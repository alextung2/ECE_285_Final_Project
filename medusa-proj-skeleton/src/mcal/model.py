"""Load the frozen Vicuna-7B v1.3 backbone + Medusa-1 heads.

Requires: torch, transformers, and the Medusa repo installed
    pip install git+https://github.com/FasterDecoding/Medusa.git
(or clone it and `pip install -e .`).

API notes, checked against FasterDecoding/Medusa (medusa/model/medusa_model.py):
  * MedusaModel.from_pretrained(path, torch_dtype=..., device_map=...) handles
    the v0.1 checkpoint format used by FasterDecoding/medusa-vicuna-7b-v1.3.
  * model.medusa            -> number of heads H
  * model.medusa_head[i]    -> head i (applied to last-layer hidden states)
  * model(input_ids, output_orig=True, medusa_forward=True)
        -> (medusa_logits [H, B, L, V], outputs, orig_logits [B, L, V])
  * model.base_model.model.medusa_mask must be None for ordinary causal forwards
    (medusa.model.utils.reset_medusa_mode does this).
  * Candidate trees use TOPK = 10 per head, so set `topk: 10` in the config.
"""

from __future__ import annotations


def load_medusa(cfg: dict):
    import torch
    from medusa.model.medusa_model import MedusaModel

    mcfg = cfg["model"]
    kwargs = dict(torch_dtype=getattr(torch, mcfg.get("dtype", "float16")),
                  low_cpu_mem_usage=True, device_map=mcfg.get("device_map", "auto"))
    if mcfg.get("load_in_4bit", False):
        # NOTE: 4-bit changes the backbone's greedy outputs. That is fine as long as
        # traces AND online decoding both use the same quantized model; never mix.
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    model = MedusaModel.from_pretrained(mcfg["name"], **kwargs)
    model.eval()
    for p in model.parameters():       # backbone AND heads stay frozen
        p.requires_grad_(False)
    tokenizer = model.get_tokenizer()
    return model, tokenizer


def gpu_report() -> str:
    """Week 1: 'GPU check'. Prints what you have before trying to load 7B."""
    try:
        import torch
    except ImportError:
        return "torch not installed"
    if not torch.cuda.is_available():
        return "CUDA not available (CPU only) - use synthetic traces or course compute"
    lines = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        lines.append(f"GPU {i}: {p.name}, {p.total_memory / 2**30:.1f} GiB")
    lines.append("Rule of thumb: Vicuna-7B fp16 ~14 GiB weights (+ heads ~1 GiB, + KV cache); "
                 "4-bit ~4-5 GiB.")
    return "\n".join(lines)
