#!/usr/bin/env python3
"""Inspect a saved mismatch without changing calibration traces or settings."""

import argparse
import json
import runpy
from pathlib import Path


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


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
    reference = record["reference_token_ids"]
    saved = record["generated_token_ids"]
    position = first_difference(reference, saved)
    if position is None or position >= min(len(reference), len(saved)):
        raise RuntimeError("This diagnostic requires a differing token, not only a length difference.")

    loaded = runpy.run_path(str(args.project_dir / "load_model.py"))
    model = loaded["model"]
    model.requires_grad_(False)

    import torch
    import medusa.model.medusa_model as medusa_module
    from fastchat.conversation import get_conv_template
    from medusa.model.kv_cache import initialize_past_key_values
    from medusa.model.utils import reset_medusa_mode

    tokenizer = model.get_tokenizer()
    conversation = get_conv_template("vicuna_v1.1")
    conversation.append_message(conversation.roles[0], record["instruction"])
    conversation.append_message(conversation.roles[1], None)
    inputs = tokenizer(conversation.get_prompt(), return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    shared = reference[:position]
    token_a, token_b = reference[position], saved[position]
    methods = {}

    def describe(name, scores):
        scores = scores.detach().float().cpu()
        values, ids = scores.topk(5)
        details = {
            "top_token_id": int(ids[0]),
            "top1_minus_top2": float(values[0] - values[1]),
            "reference_token_score": float(scores[token_a]),
            "saved_medusa_token_score": float(scores[token_b]),
            "reference_minus_medusa_score": float(scores[token_a] - scores[token_b]),
            "top5": [
                {"token_id": int(i), "text": tokenizer.decode([int(i)]), "score": float(v)}
                for i, v in zip(ids, values)
            ],
        }
        methods[name] = details
        return details

    print("\nMISMATCH REPLAY START", flush=True)
    print("Prompt ID:", args.prompt_id, flush=True)
    print("First differing token, one-based:", position + 1, flush=True)

    with torch.inference_mode():
        # Compute both modes on the exact same known prefix.
        reset_medusa_mode(model)
        prefix = torch.cat([
            inputs.input_ids,
            torch.tensor([shared], dtype=torch.long, device=model.device),
        ], dim=1)
        full_logits = model(input_ids=prefix, use_cache=False).logits[0, -1]
        describe("full_prefix_without_cache", full_logits)
        del full_logits

        (model.past_key_values, model.past_key_values_data,
         model.current_length_data) = initialize_past_key_values(model)
        cached_logits = model(
            **inputs, past_key_values=model.past_key_values, use_cache=True,
        ).logits[0, -1]
        for token_id in shared:
            cached_logits = model(
                input_ids=torch.tensor([[token_id]], device=model.device),
                past_key_values=model.past_key_values, use_cache=True,
            ).logits[0, -1]
        describe("token_by_token_cache_same_prefix", cached_logits)
        del cached_logits

        # Run the repository's original Medusa generator and record its token IDs.
        original_candidates = medusa_module.generate_candidates
        original_update = medusa_module.update_inference_inputs
        capture = {"root_logits": None, "target": None}
        official_tokens = []

        def record_candidates(medusa_logits, logits, *call_args, **kwargs):
            capture["root_logits"] = logits[0, -1].detach().clone()
            return original_candidates(medusa_logits, logits, *call_args, **kwargs)

        def record_update(*call_args, **kwargs):
            before = call_args[0].shape[1] - input_length
            best = int(call_args[2].item())
            accepted = int(call_args[3].item())
            result = original_update(*call_args, **kwargs)
            tokens = result[0][0, input_length:].tolist()
            offset = position - before
            if 0 <= offset <= accepted:
                scores = (
                    capture["root_logits"] if offset == 0
                    else call_args[6][best, offset - 1]
                )
                details = describe("official_medusa_at_difference", scores)
                capture["target"] = {
                    "same_shared_prefix": tokens[:position] == shared,
                    "offset_in_verification_round": offset,
                    "prediction_source": "root_prediction" if offset == 0 else "tree_verification",
                    "top_token_id": details["top_token_id"],
                }
            official_tokens[:] = trim_tokens(tokens, len(reference), tokenizer.eos_token_id)
            return result

        medusa_module.generate_candidates = record_candidates
        medusa_module.update_inference_inputs = record_update
        stream = None
        try:
            reset_medusa_mode(model)
            stream = model.medusa_generate(
                inputs.input_ids, temperature=0.0, max_steps=len(reference),
            )
            for _ in stream:
                if len(official_tokens) >= len(reference) or tokenizer.eos_token_id in official_tokens:
                    break
        finally:
            if stream is not None:
                stream.close()
            medusa_module.generate_candidates = original_candidates
            medusa_module.update_inference_inputs = original_update
            reset_medusa_mode(model)

    report = {
        "prompt_id": args.prompt_id,
        "instruction": record["instruction"],
        "first_difference_one_based": position + 1,
        "shared_prefix": tokenizer.decode(shared),
        "reference_token_id": token_a,
        "saved_medusa_token_id": token_b,
        "official_matches_saved_collector": official_tokens == saved,
        "official_matches_uncached_reference": official_tokens == reference,
        "official_first_difference_from_reference_one_based": (
            None if first_difference(reference, official_tokens) is None
            else first_difference(reference, official_tokens) + 1
        ),
        "official_prediction_context": capture["target"],
        "generation_config": model.generation_config.to_diff_dict(),
        "score_comparisons": methods,
        "official_token_ids": official_tokens,
        "reference_continuation": tokenizer.decode(reference[position:position + 16]),
        "official_continuation": tokenizer.decode(official_tokens[position:position + 16]),
    }
    diagnostics = args.project_dir / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    output_path = diagnostics / f"prompt_{args.prompt_id}_replay.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nREPLAY SUMMARY", flush=True)
    print("Official Medusa matches saved collector:", report["official_matches_saved_collector"], flush=True)
    print("Official Medusa matches uncached reference:", report["official_matches_uncached_reference"], flush=True)
    print("Official prediction context:", report["official_prediction_context"], flush=True)
    print("Reference token:", repr(tokenizer.decode([token_a])), flush=True)
    print("Saved Medusa token:", repr(tokenizer.decode([token_b])), flush=True)
    for name, details in methods.items():
        print(f"\n{name}:", flush=True)
        print("Top-1 token:", repr(tokenizer.decode([details["top_token_id"]])), flush=True)
        print("Reference-minus-Medusa score:", details["reference_minus_medusa_score"], flush=True)
        print("Top-1 minus top-2:", details["top1_minus_top2"], flush=True)
        print("Top-5:", details["top5"], flush=True)
    print("\nGeneration config:", report["generation_config"], flush=True)
    print("Saved diagnostic report:", output_path, flush=True)
    print("MISMATCH REPLAY END", flush=True)


if __name__ == "__main__":
    main()
