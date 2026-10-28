#!/usr/bin/env python3
"""Collect node-level Medusa calibration traces using the saved Colab loader.

Run with the project's Python 3.11 interpreter:
    python collect_medusa_pilot.py --project-dir PATH --num-prompts 1

Features are obtained before verification. Labels are computed after verification.
The root is a backbone greedy prediction and is excluded from node-level rows.
This instrumented collection program is not a latency benchmark.
"""

import argparse
import hashlib
import json
import math
import runpy
import subprocess
from importlib.metadata import version
from pathlib import Path


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def node_locations(retrieval_rows, paths):
    locations = {}
    for row_index, row in enumerate(retrieval_rows):
        for depth, node_id in enumerate(row):
            if node_id > 0:
                locations.setdefault(node_id, (row_index, depth))
    assert set(locations) == set(range(1, len(paths) + 1)), "Incomplete node mapping"
    assert all(locations[i][1] == len(path) for i, path in enumerate(paths, 1))
    return locations


def make_node_rows(paths, locations, probabilities, candidates, matches,
                   prompt_id, round_id, remaining, eos_token_id):
    """Convert verified paths into unique node records, including failed prefixes."""
    path_to_id = {tuple(path): i for i, path in enumerate(paths, 1)}
    rows = []
    for node_id, path in enumerate(paths, 1):
        depth = len(path)
        row_index, position = locations[node_id]
        assert position == depth
        candidate_path = candidates[row_index]

        # Exclude extensions beyond EOS or the requested generation length.
        if depth + 1 > remaining or eos_token_id in candidate_path[:depth]:
            continue

        parent_accepted = int(all(matches[row_index][:depth - 1]))
        prefix_accepted = int(parent_accepted and matches[row_index][depth - 1])
        confidences = [float(probabilities[h][rank]) for h, rank in enumerate(path)]
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in confidences)
        rows.append({
            "prompt_id": prompt_id,
            "round": round_id,
            "node_id": node_id,
            "parent_node_id": path_to_id[tuple(path[:-1])] if depth > 1 else 0,
            "head": depth,
            "depth": depth,
            "rank": path[-1] + 1,
            "parent_rank": path[-2] + 1 if depth > 1 else 0,
            "rank_path": [rank + 1 for rank in path],
            "token_id": candidate_path[depth],
            "confidence": confidences[-1],
            "parent_confidence": confidences[-2] if depth > 1 else 1.0,
            "marginal_product": math.prod(confidences),
            "parent_accepted": parent_accepted,
            "prefix_accepted": prefix_accepted,
            "conditional_label": prefix_accepted if parent_accepted else None,
        })

    # Check that every child's parent label agrees with the parent's own record.
    by_id = {row["node_id"]: row for row in rows}
    for row in rows:
        parent_id = row["parent_node_id"]
        if parent_id:
            assert parent_id in by_id, "A recorded node is missing its parent"
            assert row["parent_accepted"] == by_id[parent_id]["prefix_accepted"]
        assert row["prefix_accepted"] <= row["parent_accepted"]
    return rows


def summarize(payloads):
    completed = [item for item in payloads if item["status"] == "ok"]
    rows = [row for item in completed for row in item["nodes"]]
    return {
        "completed_prompts": len(completed),
        "skipped_prompts": sum(item["status"] == "skipped_context" for item in payloads),
        "mismatched_prompts": sum(item["status"] == "token_mismatch" for item in payloads),
        "candidate_node_records": len(rows),
        "conditional_training_records": sum(row["parent_accepted"] for row in rows),
        "accepted_prefix_records": sum(row["prefix_accepted"] for row in rows),
        "failed_prefix_records": sum(1 - row["prefix_accepted"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=1)
    args = parser.parse_args()
    project = args.project_dir
    loader = project / "load_model.py"
    prompt_file = project / "pilot_prompts.json"
    prompts = json.loads(prompt_file.read_text(encoding="utf-8"))
    if not 1 <= args.num_prompts <= len(prompts):
        parser.error("--num-prompts must be between 1 and the number of pilot prompts")
    prompts = prompts[:args.num_prompts]
    max_new_tokens = 64

    # Execute the exact loader that already passed the Colab loading checks.
    loaded = runpy.run_path(str(loader))
    model = loaded["model"]
    model.requires_grad_(False)
    model_id = loaded["model_id"]

    import torch
    import medusa.model.medusa_model as medusa_module
    from fastchat.conversation import get_conv_template
    from medusa.model.kv_cache import initialize_past_key_values
    from medusa.model.utils import (
        TOPK, generate_candidates, generate_medusa_buffers, initialize_medusa,
        tree_decoding, evaluate_posterior, update_inference_inputs, reset_medusa_mode,
    )

    tokenizer = model.get_tokenizer()
    eos = tokenizer.eos_token_id
    choices = [list(path) for path in model.get_medusa_choice(model.base_model_name_or_path)]
    paths = sorted(choices, key=lambda path: (len(path), path))
    buffers = generate_medusa_buffers(choices, device=model.device)
    locations = node_locations(buffers["retrieve_indices"].cpu().tolist(), paths)
    tree_budget = len(paths) + 1
    repo_dir = Path(medusa_module.__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    settings = {
        "schema_version": 1,
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "loader_sha256": hashlib.sha256(loader.read_bytes()).hexdigest(),
        "pilot_prompts_sha256": hashlib.sha256(prompt_file.read_bytes()).hexdigest(),
        "dataset_revision": json.loads((project / "data_manifest.json").read_text())["revision"],
        "model_id": model_id,
        "medusa_git_commit": commit,
        "package_versions": {name: version(name) for name in ["torch", "transformers", "bitsandbytes"]},
        "temperature": 0.0,
        "max_new_tokens": max_new_tokens,
        "tree_budget_including_root": tree_budget,
        "tree_choices_zero_based": choices,
        "ranks_in_records": "one-based; parent_rank=0 denotes the backbone root",
        "timing_benchmark": False,
    }
    output_dir = project / "pilot_node_traces_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings_path = output_dir / "collection_settings.json"
    if settings_path.exists():
        assert json.loads(settings_path.read_text()) == settings, "Existing trace settings differ"
    else:
        write_json(settings_path, settings)

    print("\nPILOT NODE COLLECTION START", flush=True)
    print("Candidate tree nodes, including root:", tree_budget, flush=True)
    payloads = []
    for number, prompt in enumerate(prompts, 1):
        prompt_id = prompt["prompt_id"]
        destination = output_dir / f"prompt_{prompt_id}.json"
        if destination.exists():
            payload = json.loads(destination.read_text(encoding="utf-8"))
            assert payload["prompt_id"] == prompt_id
            assert payload["instruction"] == prompt["instruction"]
            if payload["status"] == "token_mismatch":
                raise RuntimeError(f"A saved token mismatch needs inspection: {destination}")
            payloads.append(payload)
            print(f"[{number}/{len(prompts)}] Reusing saved prompt {prompt_id}", flush=True)
            continue

        conversation = get_conv_template("vicuna_v1.1")
        conversation.append_message(conversation.roles[0], prompt["instruction"])
        conversation.append_message(conversation.roles[1], None)
        inputs = tokenizer(conversation.get_prompt(), return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]
        payload = {"prompt_id": prompt_id, "instruction": prompt["instruction"],
                   "input_token_count": input_length, "nodes": [], "rounds": []}

        # Tree verification temporarily stores every tree node in the KV cache.
        if input_length + max_new_tokens + tree_budget > model.config.max_position_embeddings:
            payload["status"] = "skipped_context"
            write_json(destination, payload)
            payloads.append(payload)
            print(f"[{number}/{len(prompts)}] Skipped long prompt {prompt_id}", flush=True)
            continue

        print(f"[{number}/{len(prompts)}] Collecting prompt {prompt_id}...", flush=True)
        reset_medusa_mode(model)
        with torch.inference_mode():
            # Reference output checks correctness; it is not timed.
            reference = model.generate(
                **inputs, do_sample=False, max_new_tokens=max_new_tokens,
                use_cache=False, pad_token_id=eos,
            )[0, input_length:].tolist()

            if not hasattr(model, "past_key_values"):
                (model.past_key_values, model.past_key_values_data,
                 model.current_length_data) = initialize_past_key_values(model)
            model.current_length_data.zero_()
            sequence = inputs.input_ids.clone()
            generated = []
            new_token_count = 0

            try:
                medusa_logits, logits = initialize_medusa(
                    sequence, model, buffers["medusa_attn_mask"], model.past_key_values,
                )
                for round_id in range(1, max_new_tokens + 1):
                    # Capture confidence features before the verification forward pass.
                    head_logits = medusa_logits[:, 0, -1]
                    top_ids = torch.topk(head_logits, TOPK, dim=-1).indices
                    probabilities = torch.softmax(head_logits.float(), dim=-1)
                    probabilities = probabilities.gather(-1, top_ids).cpu().tolist()
                    candidates, tree_candidates = generate_candidates(
                        medusa_logits, logits, buffers["tree_indices"],
                        buffers["retrieve_indices"], temperature=0.0,
                    )
                    next_medusa, verified_logits, outputs = tree_decoding(
                        model, tree_candidates, model.past_key_values,
                        buffers["medusa_position_ids"], sequence, buffers["retrieve_indices"],
                    )
                    matches = (
                        candidates[:, 1:] == verified_logits[:, :-1].argmax(dim=-1)
                    ).cpu().tolist()
                    candidate_rows = candidates.cpu().tolist()
                    payload["nodes"].extend(make_node_rows(
                        paths, locations, probabilities, candidate_rows, matches,
                        prompt_id, round_id, max_new_tokens - len(generated), eos,
                    ))
                    best, accepted = evaluate_posterior(verified_logits, candidates, 0.0)
                    selected = candidate_rows[int(best.item())][:int(accepted.item()) + 1]
                    used = selected[:max_new_tokens - len(generated)]
                    if eos in used:
                        used = used[:used.index(eos) + 1]
                    generated.extend(used)
                    payload["rounds"].append({
                        "round": round_id, "raw_accepted": len(selected), "used_tokens": len(used),
                    })
                    sequence, logits, medusa_logits, new_token_count = update_inference_inputs(
                        sequence, candidates, best, accepted, buffers["retrieve_indices"],
                        outputs, verified_logits, next_medusa, new_token_count,
                        model.past_key_values_data, model.current_length_data,
                    )
                    if len(generated) >= max_new_tokens or eos in generated:
                        break
            finally:
                reset_medusa_mode(model)

        matched = generated == reference
        payload.update({
            "status": "ok" if matched else "token_mismatch",
            "exact_token_match": matched,
            "generated_token_ids": generated,
            "reference_token_ids": reference,
            "output": tokenizer.decode(generated, skip_special_tokens=True),
        })
        write_json(destination, payload)
        payloads.append(payload)
        eligible = sum(row["parent_accepted"] for row in payload["nodes"])
        print(f"  Saved {len(payload['nodes'])} node records; "
              f"conditional records={eligible}; exact match={matched}", flush=True)
        if not matched:
            raise RuntimeError(f"Token mismatch saved for inspection: {destination}")

    summary = summarize(payloads)
    write_json(output_dir / f"summary_first_{args.num_prompts}.json", summary)
    print("\nPILOT COLLECTION SUMMARY", flush=True)
    for key, value in summary.items():
        print(f"{key}: {value}", flush=True)
    example = next((row for item in payloads for row in item["nodes"]), None)
    if example is not None:
        print("\nExample candidate record:", flush=True)
        print(json.dumps(example, ensure_ascii=False, indent=2), flush=True)
    print("Saved directory:", output_dir, flush=True)
    print("PILOT NODE COLLECTION END", flush=True)


if __name__ == "__main__":
    main()
