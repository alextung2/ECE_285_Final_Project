#!/usr/bin/env python3
"""Try isolated precision configurations on one saved mismatch.

The original loader, collector, and traces are preserved. A passing configuration
is saved as a candidate loader and still needs validation on the full pilot set.
Quantized backbone weights are not dequantized or requantized by these patches.
"""

import argparse
import gc
import json
import runpy
from pathlib import Path


LM_HEAD_PATCH = '''
# Compute the backbone output projection in FP32.
model.lm_head.float()
if not getattr(model.lm_head, "_fp32_input_hook_installed", False):
    def _cast_lm_head_input_to_fp32(module, args):
        return (args[0].float(),) + args[1:]
    model.lm_head.register_forward_pre_hook(_cast_lm_head_input_to_fp32)
    model.lm_head._fp32_input_hook_installed = True
'''

FP32_EXECUTION_PATCH = '''
# Preserve packed 4-bit parameters and promote other floating-point tensors.
import bitsandbytes as bnb
for _module in model.modules():
    for _parameter in _module.parameters(recurse=False):
        if isinstance(_parameter, bnb.nn.Params4bit):
            continue
        if _parameter.is_floating_point() and _parameter.dtype != torch.float32:
            _parameter.data = _parameter.data.float()
    for _name, _buffer in list(_module.named_buffers(recurse=False)):
        if _buffer.is_floating_point() and _buffer.dtype != torch.float32:
            setattr(_module, _name, _buffer.float())
    if isinstance(_module, bnb.nn.Linear4bit):
        _module.compute_dtype = torch.float32
        _module.compute_type_is_set = True
'''


def first_difference(left, right):
    return next((i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
                None if len(left) == len(right) else min(len(left), len(right)))


def trim_tokens(tokens, limit, eos):
    tokens = tokens[:limit]
    return tokens[:tokens.index(eos) + 1] if eos in tokens else tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--prompt-id", type=int, default=24)
    args = parser.parse_args()
    source = args.project_dir / "pilot_node_traces_v1" / f"prompt_{args.prompt_id}.json"
    record = json.loads(source.read_text(encoding="utf-8"))
    original_loader = args.project_dir / "load_model.py"
    loaded = runpy.run_path(str(original_loader))
    model = loaded["model"]
    model.requires_grad_(False)

    import torch
    import bitsandbytes as bnb
    import medusa.model.medusa_model as medusa_module
    from fastchat.conversation import get_conv_template
    from medusa.model.utils import reset_medusa_mode

    tokenizer = model.get_tokenizer()
    conversation = get_conv_template("vicuna_v1.1")
    conversation.append_message(conversation.roles[0], record["instruction"])
    conversation.append_message(conversation.roles[1], None)
    inputs = tokenizer(conversation.get_prompt(), return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    max_new_tokens = 64
    reports = []
    selected_mode = None
    candidate_path = None
    patches = [
        ("fp32_lm_head", LM_HEAD_PATCH),
        ("4bit_weights_fp32_execution", LM_HEAD_PATCH + FP32_EXECUTION_PATCH),
    ]

    # Verify that the precision patches preserve all packed weight tensors.
    packed_weights = [
        (parameter, parameter.data_ptr(), parameter.dtype, tuple(parameter.shape))
        for parameter in model.parameters() if isinstance(parameter, bnb.nn.Params4bit)
    ]
    assert packed_weights, "Expected packed 4-bit backbone parameters"
    print("\nPRECISION TEST START", flush=True)
    print("Prompt ID:", args.prompt_id, flush=True)

    for mode, patch in patches:
        reset_medusa_mode(model)
        for attribute in ("past_key_values", "past_key_values_data", "current_length_data"):
            if hasattr(model, attribute):
                delattr(model, attribute)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        exec(patch, {"model": model, "torch": torch, "bnb": bnb})
        for parameter, pointer, dtype, shape in packed_weights:
            assert parameter.data_ptr() == pointer
            assert parameter.dtype == dtype and tuple(parameter.shape) == shape
        model.eval()
        print("\nTesting precision mode:", mode, flush=True)

        with torch.inference_mode():
            reference = model.generate(
                **inputs, do_sample=False, use_cache=False,
                max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id,
            )[0, input_length:].tolist()

            official_tokens = []
            original_update = medusa_module.update_inference_inputs

            def record_update(*call_args, **kwargs):
                result = original_update(*call_args, **kwargs)
                official_tokens[:] = trim_tokens(
                    result[0][0, input_length:].tolist(), max_new_tokens,
                    tokenizer.eos_token_id,
                )
                return result

            medusa_module.update_inference_inputs = record_update
            stream = None
            try:
                stream = model.medusa_generate(
                    inputs.input_ids, temperature=0.0, max_steps=max_new_tokens,
                )
                for _ in stream:
                    if len(official_tokens) >= max_new_tokens or tokenizer.eos_token_id in official_tokens:
                        break
            finally:
                if stream is not None:
                    stream.close()
                medusa_module.update_inference_inputs = original_update
                reset_medusa_mode(model)

        difference = first_difference(reference, official_tokens)
        matched = difference is None
        result = {
            "precision_mode": mode,
            "exact_token_match": matched,
            "reference_token_count": len(reference),
            "medusa_token_count": len(official_tokens),
            "first_difference_one_based": None if difference is None else difference + 1,
            "reference_token_ids": reference,
            "medusa_token_ids": official_tokens,
            "reference_output": tokenizer.decode(reference, skip_special_tokens=True),
            "medusa_output": tokenizer.decode(official_tokens, skip_special_tokens=True),
            "peak_gpu_tensor_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "packed_4bit_weights_preserved": True,
        }
        reports.append(result)
        print("Exact token match:", matched, flush=True)
        print("First differing position:", result["first_difference_one_based"], flush=True)
        print(f"Peak GPU tensor memory: {result['peak_gpu_tensor_memory_gib']:.2f} GiB", flush=True)

        if matched:
            selected_mode = mode
            candidate_path = args.project_dir / f"load_model_candidate_{mode}.py"
            loader_text = original_loader.read_text(encoding="utf-8")
            loader_text += "\n" + patch
            loader_text += f"\nprecision_mode = {mode!r}\n"
            loader_text += "model.eval()\nmodel.requires_grad_(False)\n"
            loader_text += "print('Candidate precision mode:', precision_mode, flush=True)\n"
            candidate_path.write_text(loader_text, encoding="utf-8")
            break

    report = {
        "prompt_id": args.prompt_id,
        "instruction": record["instruction"],
        "tested_prompt_count": 1,
        "selected_precision_mode": selected_mode,
        "candidate_loader_path": str(candidate_path) if candidate_path else None,
        "requires_full_pilot_validation": True,
        "timing_benchmark": False,
        "results": reports,
    }
    diagnostic_dir = args.project_dir / "diagnostics"
    diagnostic_dir.mkdir(exist_ok=True)
    report_path = diagnostic_dir / f"prompt_{args.prompt_id}_precision_test.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nPRECISION TEST SUMMARY", flush=True)
    print("Selected precision mode:", selected_mode, flush=True)
    print("Candidate loader:", candidate_path, flush=True)
    print("Original loader and calibration traces preserved", flush=True)
    print("Full pilot validation is still required", flush=True)
    print("Saved report:", report_path, flush=True)
    print("PRECISION TEST END", flush=True)


if __name__ == "__main__":
    main()
