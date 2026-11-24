"""Collect replay-only labels on a single cached greedy trajectory.

Each state records the draft ranks seen in the same forward that chose the
next backbone token. These traces measure agreement with that continuation.
A separate online gate tests tree-shaped verification and output equivalence.
"""
import json
from pathlib import Path

import numpy as np
import torch

from _common import parse
from mcal.audit import validate_arrays
from mcal.data import vicuna_prompt
from mcal.model import load_medusa
from mcal.precision import ranked_probabilities, require_compatible
from mcal.provenance import prompt_digest, read_prompt_records
from mcal.traces import TraceSet
from medusa.model.kv_cache import initialize_past_key_values
from medusa.model.utils import reset_medusa_mode


@torch.inference_mode()
def collect(model, tok, record, limit, topk, cache, lengths):
    reset_medusa_mode(model)
    lengths.zero_()
    ids = tok([vicuna_prompt(record["instruction"])], return_tensors="pt").input_ids.to(model.device)
    if ids.shape[1] + limit > model.config.max_position_embeddings:
        raise ValueError(f"Context exceeds model capacity for {record['prompt_id']}")
    generated, probs, ranks, roots = [], [], [], []
    for _ in range(limit):
        heads, _, logits = model(ids, past_key_values=cache, output_orig=True, medusa_forward=True)
        root = logits[0, -1].float().softmax(-1)
        token = logits[0, -1].argmax()
        p, t = ranked_probabilities(heads[:, 0, -1], topk)
        roots.append(root[token])
        probs.append(p)
        ranks.append(t)
        generated.append(int(token))
        if generated[-1] == tok.eos_token_id:
            break
        ids = token.reshape(1, 1)
    n = len(generated) - model.medusa
    if n <= 0:
        return None, generated
    candidates = torch.stack(ranks[:n]).cpu().numpy()
    future = np.lib.stride_tricks.sliding_window_view(np.asarray(generated[1:]), model.medusa)
    match = candidates == future[:, :, None]
    gold = np.where(match.any(-1), match.argmax(-1), -1).astype(np.int16)
    trace = TraceSet(gold, torch.stack(probs[:n]).cpu().numpy(),
                     torch.stack(roots[:n]).cpu().numpy(), np.full(n, record["prompt_id"], np.int32),
                     np.arange(n, dtype=np.int32), dict(source="real", execution_contract=model.execution_contract,
                     label_reference="same_forward_cached_greedy", feature_source="same_forward_cached_greedy"),
                     candidates.astype(np.int32))
    validate_arrays(trace)
    return trace, generated


def main():
    _, cfg = parse(__doc__)
    model, tok = load_medusa(cfg)
    cache, _, lengths = initialize_past_key_values(model.base_model)
    root = Path(cfg["out_dir"]) / "cached_reference"
    root.mkdir(exist_ok=True)
    for split in ("calib", "eval"):
        settings = cfg["traces"][split]
        records = read_prompt_records(settings["prompt_file"])
        parts, summaries = [], []
        folder = root / split
        folder.mkdir(exist_ok=True)
        for i, record in enumerate(records):
            key = folder / str(record["prompt_id"])
            evidence = key.with_suffix(".json")
            trace_path = key.with_suffix(".npz")
            binding = dict(prompt_sha256=prompt_digest([record]), execution_contract=model.execution_contract,
                           limit=cfg["traces"]["max_new_tokens"], topk=cfg["traces"]["topk"])
            if evidence.exists():
                saved = json.loads(evidence.read_text())
                require_compatible(binding, saved["binding"])
                trace = TraceSet.load(trace_path) if saved["positions"] else None
            else:
                trace, tokens = collect(model, tok, record, binding["limit"], binding["topk"], cache, lengths)
                saved = dict(prompt_id=record["prompt_id"], generated_ids=tokens,
                             positions=trace.n_steps if trace else 0, binding=binding)
                if trace:
                    trace.save(trace_path)
                evidence.write_text(json.dumps(saved))
            if trace:
                parts.append(trace)
            summaries.append(dict(prompt_id=saved["prompt_id"], tokens=len(saved["generated_ids"]),
                                  positions=saved["positions"]))
            print(split, i + 1, "/", len(records), flush=True)
        result = TraceSet.concat(parts)
        scored = [s["prompt_id"] for s in summaries if s["positions"]]
        result.meta.update(dataset=settings["dataset"], dataset_revision=settings["dataset_revision"],
                           collection_complete=True, expected_prompt_ids=scored, split=split,
                           input_prompt_sha256=prompt_digest(records),
                           prompt_sha256=prompt_digest([r for r in records if r["prompt_id"] in set(scored)]),
                           coverage=summaries)
        result.save(root / (split + ".npz"))
    manifest = json.loads(Path(cfg["estimators"]["manifest"]).read_text(encoding="utf-8"))
    scored = set(TraceSet.load(root / "calib.npz").prompt_id.tolist())
    for key in ("train_prompt_ids", "validation_prompt_ids"):
        manifest[key] = [pid for pid in manifest[key] if pid in scored]
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
