#!/usr/bin/env python3
"""Work-plan pilot collector: NF4/FP16, complete top-10, Medusa sequence labels.

This is an instrumented correctness/data run, never a speed benchmark.
All rounds are checkpointed in JSON. The unchanged team TraceSet interface has
no censoring mask, so NPZ exports include only rounds with all H future targets
observed in the final Medusa sequence. Missing targets are NOT negative labels.
Rows sample the default tree's visited rounds, not every generated position.
"""
import argparse
import hashlib
import json
import platform
import runpy
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mismatch_report(reference, generated, reference_scores, generated_scores):
    differences = []
    shared = True
    for i in range(max(len(reference), len(generated))):
        ref = reference[i] if i < len(reference) else None
        med = generated[i] if i < len(generated) else None
        if ref != med:
            differences.append({
                "position_zero_based": i,
                "same_generated_prefix": shared,
                "reference_token_id": ref, "medusa_token_id": med,
                "reference_scores": reference_scores[i] if ref is not None else None,
                "medusa_scores": generated_scores[i] if med is not None else None,
            })
            shared = False
    return {
        "exact_token_match": not differences,
        "first_difference_zero_based": differences[0]["position_zero_based"] if differences else None,
        "differing_positions": len(differences),
        "position_mismatch_fraction": len(differences) / max(len(reference), len(generated), 1),
        "note": "After the first divergence, the two outputs have different contexts; later gaps are descriptive only.",
        "differences": differences,
    }


def derive_rows(record, choices, n_heads=5, topk=10):
    """Derive labels from the final accepted sequence, never the plain reference."""
    generated = record["generated_token_ids"]
    rows, checks = [], []
    for round_record in record["rounds"]:
        position = round_record["position"]
        tokens = np.asarray(round_record["topk_token"], dtype=np.int32)
        probabilities = np.asarray(round_record["topk_prob"], dtype=np.float32)
        assert tokens.shape == probabilities.shape == (n_heads, topk)
        assert all(len(set(head)) == topk for head in tokens.tolist())
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        assert (probabilities.sum(axis=-1) <= 1.00001).all()
        assert (np.diff(probabilities, axis=-1) <= 1e-7).all()
        assert generated[position] == round_record["root_token_id"]
        assert 0 <= round_record["root_prob"] <= 1
        available = min(n_heads, len(generated) - position - 1)
        gold = np.full(n_heads, -1, dtype=np.int16)
        for head in range(available):
            target = generated[position + 1 + head]
            ranks = np.flatnonzero(tokens[head] == target)
            if ranks.size:
                gold[head] = ranks[0]
        simulated = 1 + sum(
            len(path) <= available and all(gold[j] == r for j, r in enumerate(path))
            for path in choices
        )
        full_horizon = available == n_heads
        checks.append({
            "round": round_record["round"], "position": position,
            "full_horizon": full_horizon,
            "online_used_tokens": round_record["used_tokens"],
            "simulated_used_tokens": simulated,
            "match": simulated == round_record["used_tokens"],
        })
        if full_horizon:
            rows.append({
                "gold_rank": gold, "topk_prob": probabilities,
                "topk_token": tokens, "root_prob": round_record["root_prob"],
                "prompt_id": record["prompt_id"], "position": position,
            })
    assert sum(r["used_tokens"] for r in record["rounds"]) == len(generated)
    return rows, checks


def export_npz(path, records, settings, complete_pilot):
    rows, checks = [], []
    for record in records:
        r, c = derive_rows(record, settings["tree_choices_zero_based"])
        rows.extend(r)
        checks.extend(c)
    if not rows:
        raise RuntimeError("No fully observed five-head rounds are available for export.")
    names = ("gold_rank", "topk_prob", "topk_token", "root_prob", "prompt_id", "position")
    dtypes = (np.int16, np.float32, np.int32, np.float32, np.int32, np.int32)
    arrays = {name: np.asarray([r[name] for r in rows], dtype=dtype)
              for name, dtype in zip(names, dtypes)}
    full_checks = [c for c in checks if c["full_horizon"]]
    summary = {
        "completed_prompts": len(records),
        "exported_prompts": len(np.unique(arrays["prompt_id"])),
        "exact_token_matches": sum(r["comparison"]["exact_token_match"] for r in records),
        "mismatched_prompts": sum(not r["comparison"]["exact_token_match"] for r in records),
        "recorded_rounds": len(checks), "exported_rounds": len(rows),
        "censored_rounds_not_exported": len(checks) - len(rows),
        "total_generated_tokens": sum(len(r["generated_token_ids"]) for r in records),
        "online_tokens_per_round_all": float(np.mean([c["online_used_tokens"] for c in checks])),
        "online_tokens_per_round_exported": float(np.mean([c["online_used_tokens"] for c in full_checks])),
        "simulated_tokens_per_round_exported": float(np.mean([c["simulated_used_tokens"] for c in full_checks])),
        "simulation_disagreements_all": sum(not c["match"] for c in checks),
        "simulation_disagreements_exported": sum(not c["match"] for c in full_checks),
        "complete_pilot": complete_pilot,
    }
    meta = dict(settings, summary=summary, prompt_ids=[r["prompt_id"] for r in records])
    arrays["meta"] = np.array(json.dumps(meta))
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    with np.load(path, allow_pickle=False) as z:
        assert z["gold_rank"].shape == (len(rows), 5)
        assert z["topk_prob"].shape == (len(rows), 5, 10)
        assert np.array_equal(z["gold_rank"], arrays["gold_rank"])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=1)
    args = parser.parse_args()
    project = args.project_dir
    loader = project / "load_model.py"
    prompt_file = project / "pilot_prompts.json"
    pilot = read_json(prompt_file)
    if not 1 <= args.num_prompts <= len(pilot):
        parser.error("Invalid pilot prompt count")
    train_ids = {p["prompt_id"] for p in read_json(project / "train_prompts.json")}
    validation_ids = {p["prompt_id"] for p in read_json(project / "validation_prompts.json")}
    assert train_ids.isdisjoint(validation_ids), "Train/validation ID overlap"
    assert len({p["prompt_id"] for p in pilot}) == len(pilot)
    assert {p["prompt_id"] for p in pilot} <= train_ids
    output_dir = project / "shared_traces_fp16_v1" / "pilot"
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import bitsandbytes as bnb
    import medusa.model.medusa_model as medusa_module
    from fastchat.conversation import get_conv_template
    from medusa.model.kv_cache import initialize_past_key_values
    from medusa.model.utils import (
        TOPK, generate_candidates, generate_medusa_buffers, initialize_medusa,
        tree_decoding, evaluate_posterior, update_inference_inputs, reset_medusa_mode,
    )

    assert torch.cuda.is_available(), "A CUDA GPU is required"
    loaded = runpy.run_path(str(loader))
    model = loaded["model"]
    model.eval().requires_grad_(False)
    assert model.dtype == torch.float16, "Use the original FP16 loader"
    assert model.lm_head.weight.dtype == torch.float16
    assert all(p.dtype == torch.float16 for p in model.medusa_head.parameters())
    assert not isinstance(model.lm_head, bnb.nn.Linear4bit)
    assert not any(isinstance(m, bnb.nn.Linear4bit) for m in model.medusa_head.modules())
    quantized = [m for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)]
    assert quantized, "Expected a 4-bit backbone"
    assert all(m.compute_dtype == torch.float16 for m in quantized)
    assert all(m.weight.quant_type == "nf4" and m.weight.compress_statistics for m in quantized)
    tokenizer = model.get_tokenizer()
    eos = tokenizer.eos_token_id
    assert model.medusa == 5 and TOPK == 10
    choices = [list(p) for p in model.get_medusa_choice(model.base_model_name_or_path)]
    buffers = generate_medusa_buffers(choices, device=model.device)
    repo_dir = Path(medusa_module.__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()
    assert commit == "e2a5d20c048a9b0a4092e6933c34313687422518", "Unexpected upstream revision"
    settings = {
        "schema_version": 1,
        "collector_sha256": sha256(Path(__file__)), "loader_sha256": sha256(loader),
        "pilot_prompts_sha256": sha256(prompt_file),
        "train_prompts_sha256": sha256(project / "train_prompts.json"),
        "validation_prompts_sha256": sha256(project / "validation_prompts.json"),
        "dataset_revision": read_json(project / "data_manifest.json")["revision"],
        "model_id": loaded["model_id"], "medusa_git_commit": commit,
        "backbone_config_commit": getattr(model.config, "_commit_hash", None),
        "python": platform.python_version(),
        "packages": {n: version(n) for n in ("torch", "transformers", "bitsandbytes", "accelerate", "fschat", "numpy")},
        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "precision": "NF4_double_quant_FP16_compute_FP16_heads",
        "conversation_template": "vicuna_v1.1", "max_new_tokens": 64,
        "temperature": 0.0, "topk": 10, "n_heads": 5,
        "tree_choices_zero_based": choices,
        "draft_budget_excluding_root": len(choices),
        "tree_nodes_including_root": len(choices) + 1,
        "label_source": "final accepted Medusa sequence, not uncached greedy",
        "sampling": "default-tree online verification-round start positions",
        "tail_policy": "Preserve all rounds in JSON; export only rounds with all five future targets observed.",
        "rank_convention": "zero-based; -1 means observed target absent from top-10",
        "position_convention": "zero-based generated index of the root token",
        "split": "pilot_subset_of_training", "timing_benchmark": False,
        "scope": "Offline scores on recorded contexts; dynamic-policy performance requires online validation.",
    }
    settings_path = output_dir / "collection_settings.json"
    if settings_path.exists():
        assert read_json(settings_path) == settings, "Existing trace settings differ; do not mix runs"
    else:
        write_json(settings_path, settings)
        frozen = subprocess.check_output(
            [__import__("sys").executable, "-m", "pip", "freeze"], text=True,
        )
        (output_dir / "environment_actual.txt").write_text(frozen, encoding="utf-8")

    def score_info(scores):
        values, ids = torch.topk(scores.float(), 2)
        return {
            "argmax_token_id": int(scores.argmax().item()),
            "top2_token_ids": ids.cpu().tolist(),
            "top2_logits": values.cpu().tolist(),
            "top1_minus_top2": float((values[0] - values[1]).item()),
        }

    records = []
    print("\nSHARED TRACE COLLECTION START", flush=True)
    print("Precision: NF4 backbone with FP16 compute and heads", flush=True)
    print("Mismatch policy: record and continue", flush=True)
    for index, prompt in enumerate(pilot[:args.num_prompts], 1):
        prompt_id = prompt["prompt_id"]
        destination = output_dir / f"prompt_{prompt_id}.json"
        if destination.exists():
            record = read_json(destination)
            assert record["instruction"] == prompt["instruction"] and record["prompt_id"] == prompt_id
            assert record["status"] == "collected"
            derive_rows(record, choices)
            records.append(record)
            print(f"[{index}/{args.num_prompts}] Reusing prompt {prompt_id}", flush=True)
            continue
        print(f"[{index}/{args.num_prompts}] Collecting prompt {prompt_id}...", flush=True)
        conversation = get_conv_template("vicuna_v1.1")
        conversation.append_message(conversation.roles[0], prompt["instruction"])
        conversation.append_message(conversation.roles[1], None)
        inputs = tokenizer(conversation.get_prompt(), return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]
        assert input_length + 64 + len(choices) + 1 <= model.config.max_position_embeddings, "Prompt exceeds the KV-cache capacity"
        reset_medusa_mode(model)
        with torch.inference_mode():
            reference_result = model.generate(
                **inputs, do_sample=False, max_new_tokens=64, use_cache=False,
                pad_token_id=eos, return_dict_in_generate=True, output_scores=True,
            )
            reference = reference_result.sequences[0, input_length:].tolist()
            reference_scores = [score_info(s[0]) for s in reference_result.scores]
            del reference_result
            if not hasattr(model, "past_key_values"):
                (model.past_key_values, model.past_key_values_data,
                 model.current_length_data) = initialize_past_key_values(model)
            model.current_length_data.zero_()
            sequence = inputs.input_ids.clone()
            generated, generated_scores, rounds = [], [], []
            new_token_count = 0
            try:
                medusa_logits, logits = initialize_medusa(sequence, model, buffers["medusa_attn_mask"], model.past_key_values)
                for round_id in range(1, 65):
                    position = len(generated)
                    head_logits = medusa_logits[:, 0, -1]
                    top_ids = torch.topk(head_logits, TOPK, dim=-1).indices
                    top_prob = head_logits.float().softmax(-1).gather(-1, top_ids)
                    root_logits = logits[0, -1]
                    root_id = int(root_logits.argmax().item())
                    root_prob = float(root_logits.float().softmax(-1)[root_id].item())
                    candidates, tree_candidates = generate_candidates(
                        medusa_logits, logits, buffers["tree_indices"], buffers["retrieve_indices"], temperature=0.0,
                    )
                    next_medusa, verified_logits, outputs = tree_decoding(
                        model, tree_candidates, model.past_key_values,
                        buffers["medusa_position_ids"], sequence, buffers["retrieve_indices"],
                    )
                    best, accepted = evaluate_posterior(verified_logits, candidates, 0.0)
                    best_index = int(best.item())
                    selected = candidates[best_index, :int(accepted.item()) + 1].tolist()
                    used = selected[:64 - position]
                    if eos in used:
                        used = used[:used.index(eos) + 1]
                    assert used[0] == root_id
                    for offset, token_id in enumerate(used):
                        scores = root_logits if offset == 0 else verified_logits[best_index, offset - 1]
                        info = score_info(scores)
                        assert info["argmax_token_id"] == token_id
                        generated_scores.append(info)
                    rounds.append({
                        "round": round_id, "position": position,
                        "root_token_id": root_id, "root_prob": root_prob,
                        "topk_token": top_ids.cpu().tolist(), "topk_prob": top_prob.cpu().tolist(),
                        "raw_accepted_tokens": len(selected), "used_tokens": len(used),
                    })
                    generated.extend(used)
                    sequence, logits, medusa_logits, new_token_count = update_inference_inputs(
                        sequence, candidates, best, accepted, buffers["retrieve_indices"],
                        outputs, verified_logits, next_medusa, new_token_count,
                        model.past_key_values_data, model.current_length_data,
                    )
                    if len(generated) >= 64 or eos in generated:
                        break
            finally:
                reset_medusa_mode(model)
        record = {
            "status": "collected", "prompt_id": prompt_id, "instruction": prompt["instruction"],
            "input_token_count": input_length, "generated_token_ids": generated,
            "reference_token_ids": reference, "rounds": rounds,
            "medusa_output": tokenizer.decode(generated, skip_special_tokens=True),
            "reference_output": tokenizer.decode(reference, skip_special_tokens=True),
            "comparison": mismatch_report(reference, generated, reference_scores, generated_scores),
        }
        rows, checks = derive_rows(record, choices)
        record["simulation_checks"] = checks
        write_json(destination, record)
        records.append(record)
        print(f"  Rounds={len(rounds)}; exported rows={len(rows)}; exact match={record['comparison']['exact_token_match']}", flush=True)

    complete = args.num_prompts == len(pilot)
    npz_path = output_dir / ("calib.npz" if complete else f"calib_first_{args.num_prompts}.npz")
    summary = export_npz(npz_path, records, settings, complete)
    write_json(output_dir / f"summary_first_{args.num_prompts}.json", summary)
    print("\nSHARED TRACE SUMMARY", flush=True)
    for key, value in summary.items():
        print(f"{key}: {value}", flush=True)
    with np.load(npz_path, allow_pickle=False) as z:
        print("gold_rank shape:", z["gold_rank"].shape, flush=True)
        print("topk_prob shape:", z["topk_prob"].shape, flush=True)
    print("Saved TraceSet:", npz_path, flush=True)
    print("SHARED TRACE COLLECTION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
