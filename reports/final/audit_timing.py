"""Recompute throughput and exact agreements from the saved timed generations."""
import argparse
import csv
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("results", type=Path)
args = parser.parse_args()
for budget in (16, 63):
    path = args.results / f"speed_B{budget}"
    records = json.loads(path.with_name(path.name+"_prompts.json").read_text())
    with path.with_suffix(".csv").open(newline="") as source:
        summary = list(csv.DictReader(source))
    assert len(records) == 48 and len(summary) == 6
    by_method = {row["method"]: [r for r in records if r["method"] == row["method"]] for row in summary}
    greedy = {r["prompt_id"]: r["generated_ids"] for r in by_method["greedy"]}
    baseline = sum(map(len, greedy.values())) / sum(r["wall_s"] for r in by_method["greedy"])
    for row in summary:
        pairs = by_method[row["method"]]
        assert sorted(r["prompt_id"] for r in pairs) == list(range(8))
        tokens = sum(len(r["generated_ids"]) for r in pairs)
        assert all(len(r["generated_ids"]) == r["new_tokens"] and r["wall_s"] > 0 for r in pairs)
        matches = [r["generated_ids"] == greedy[r["prompt_id"]] for r in pairs]
        assert matches == [r["agree"] for r in pairs]
        tps = tokens / sum(r["wall_s"] for r in pairs)
        for key, value in [("tokens_per_s",tps), ("speedup",tps/baseline), ("agree",sum(matches)/8),
                           ("accepted",tokens/sum(r["steps"] for r in pairs))]:
            assert math.isclose(float(row[key]),value,abs_tol=1e-10), (budget,row["method"],key)
    print(f"B={budget}: all 48 timed records reproduce their reported metrics")
