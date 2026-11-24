"""Independently reconstruct every replay target from per-prompt token IDs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", type=Path, help="Prepared experiment directory")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
result = {}
for split in ("calib", "eval"):
    reference = args.run / "cached_reference"
    records = json.loads((args.run / "prompts" / (split + "_full.json")).read_text(encoding="utf-8"))
    archive = reference / (split + ".npz")
    counts, short = [], []
    with np.load(archive, allow_pickle=False) as a:
        assert set(np.unique(a["prompt_id"])).issubset({r["prompt_id"] for r in records})
        all_ids, all_pos = a["prompt_id"], a["position"]
        all_ranks, all_tokens = a["gold_rank"], a["topk_token"]
        for row in records:
            pid = row["prompt_id"]
            evidence = json.loads((reference / split / f"{pid}.json").read_text())
            assert pid == evidence["prompt_id"]
            generated = np.asarray(evidence["generated_ids"])
            where = np.flatnonzero(all_ids == pid)
            n = max(0, len(generated) - all_ranks.shape[1])
            np.testing.assert_array_equal(all_pos[where], np.arange(n))
            assert evidence["positions"] == n
            if n == 0:
                short.append(pid)
            for head in range(all_ranks.shape[1]):
                hits = all_tokens[where, head] == generated[head+1:head+1+n, None]
                measured = np.where(hits.any(axis=1), hits.argmax(axis=1), -1)
                np.testing.assert_array_equal(measured, all_ranks[where, head])
            counts.append(n)
        assert sum(counts) == len(all_ids)
    result[split] = dict(input_prompts=len(records), scored_prompts=len(records)-len(short),
                         positions=sum(counts), zero_horizon_ids=short, verified_head_labels=sum(counts)*5,
                         archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
args.out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
