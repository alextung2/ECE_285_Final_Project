"""Explicit execution contracts; mode names never stand in for runtime evidence."""
from __future__ import annotations

MODES = ("nf4_fp16", "nf4_fp32_from_fp16")


def precision_spec(mcfg):
    mode = mcfg.get("precision_mode", "nf4_fp16")
    if mode not in MODES:
        raise ValueError(f"Unknown precision_mode {mode!r}; choose {MODES}")
    if not mcfg.get("load_in_4bit", False) or mcfg.get("dtype", "float16") != "float16":
        raise ValueError("These modes require load_in_4bit=true and initial dtype=float16")
    dtype = "float16" if mode == "nf4_fp16" else "float32"
    return dict(schema_version=1, model=mcfg["name"], precision_mode=mode,
                weight_storage="nf4", double_quant=True, initial_dtype="float16",
                compute_dtype=dtype, head_dtype=dtype,
                lm_head_fp32_input_hook=(dtype == "float32"),
                promotion="none" if mode == "nf4_fp16" else "floating_parameters_and_buffers",
                tf32=False, rank_order="raw_logits_topk", probability_dtype="float32",
                prompt_template="vicuna_v1.1", temperature=0.0)


def promote_execution(model, dtype, linear4bit, params4bit):
    """Promote floating state without touching packed weights or quantization state.

    Deliberately load/quantize in FP16 first, keeping the quantized model identical
    between arms. Matches Lucia's LM_HEAD_PATCH + FP32_EXECUTION_PATCH committed
    in 54cfc8c, including the LM-head input hook. TF32 policy is set separately
    by our loader; her historical loader leaves it to the environment.
    Do not call model.float() on a quantized model.
    """
    import torch
    with torch.no_grad():
        if dtype == torch.float32:
            model.lm_head.float()
            if not getattr(model.lm_head, "_fp32_input_hook_installed", False):
                def cast_lm_head_input_to_fp32(module, args):
                    return (args[0].float(),) + args[1:]
                model.lm_head.register_forward_pre_hook(cast_lm_head_input_to_fp32)
                model.lm_head._fp32_input_hook_installed = True
        for p in model.parameters():
            if not isinstance(p, params4bit) and p.is_floating_point():
                p.data = p.data.to(dtype)
        for module in model.modules():
            for name, buf in module.named_buffers(recurse=False):
                if buf.is_floating_point():
                    setattr(module, name, buf.to(dtype))
            if isinstance(module, linear4bit):
                module.compute_dtype = dtype
                module.compute_type_is_set = True


def audit_model(model, spec, linear4bit, params4bit):
    import torch
    expected = getattr(torch, spec["compute_dtype"])
    hook_installed = bool(getattr(model.lm_head, "_fp32_input_hook_installed", False))
    if hook_installed != spec["lm_head_fp32_input_hook"]:
        raise RuntimeError("LM-head input hook differs from precision contract")
    layers = [m for m in model.modules() if isinstance(m, linear4bit)]
    if not layers or any(m.compute_dtype != expected for m in layers):
        raise RuntimeError("Missing 4-bit layers or incorrect computation dtype")
    if any(m.weight.dtype != torch.uint8 or m.weight.quant_type != "nf4"
           or not m.weight.compress_statistics or m.weight.quant_state is None for m in layers):
        raise RuntimeError("Expected initialized packed NF4 double-quantized weights")
    for head in (model.lm_head, model.medusa_head):
        if any(isinstance(m, linear4bit) for m in head.modules()):
            raise RuntimeError("Prediction heads must not be quantized")
        if any(p.dtype != expected for p in head.parameters()):
            raise RuntimeError("Prediction-head dtype differs from precision contract")
    if any(p.dtype != expected for p in model.parameters()
           if p.is_floating_point() and not isinstance(p, params4bit)):
        raise RuntimeError("Floating model parameter dtype differs from precision contract")
    buffer_dtypes = sorted({str(b.dtype) for b in model.buffers() if b.is_floating_point()})
    if expected == torch.float32 and any(d != str(torch.float32) for d in buffer_dtypes):
        raise RuntimeError("FP32 execution requires FP32 floating buffers")
    return dict(linear4bit_count=len(layers), compute_dtype=str(expected),
                lm_head_fp32_input_hook=hook_installed,
                floating_buffer_dtypes=buffer_dtypes,
                lm_head_dtype=str(model.lm_head.weight.dtype),
                medusa_head_dtypes=sorted({str(p.dtype) for p in model.medusa_head.parameters()}))


def require_compatible(left, right):
    """Reject unknown provenance as well as known mismatches; no inferred FP16."""
    if not left or not right or left != right:
        raise ValueError("Incompatible or missing execution contract; recollect/refit, do not relabel historical data")


def trace_contract(ts):
    if ts.meta.get("source") == "synthetic":
        return {"source": "synthetic", "n_heads": ts.n_heads, "topk": ts.topk}
    contract = ts.meta.get("execution_contract")
    if not contract:
        raise ValueError("Real traces have no execution contract; historical files are read-only evidence")
    return dict(contract, n_heads=ts.n_heads, topk=ts.topk)


def require_trace_config(ts, cfg, synthetic=False):
    contract = trace_contract(ts)
    if synthetic:
        if contract.get("source") != "synthetic":
            raise ValueError("--synthetic requires synthetic traces")
    else:
        expected = precision_spec(cfg["model"])
        if any(contract.get(k) != v for k, v in expected.items()):
            raise ValueError("Trace precision/configuration mismatch")
    return contract


def ranked_probabilities(logits, topk):
    """Use upstream candidate ranks, then gather full-vocabulary FP32 probabilities."""
    import torch
    ids = logits.topk(topk, dim=-1).indices
    return torch.softmax(logits.float(), dim=-1).gather(-1, ids), ids
