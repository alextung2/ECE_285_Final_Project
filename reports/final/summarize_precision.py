"""Summarize strict precision logs while retaining the original failure reports."""
import argparse
import ast
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("known", type=Path)
parser.add_argument("pilot", type=Path)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
output = {}
for group,path in [("known", args.known), ("archived_failed_pilot_subset", args.pilot)]:
    report = json.loads(path.read_text())
    rows = []
    for entry in report["prompts"]:
        errors = []
        if entry.get("cached_uncached_agree") is False:
            errors.append("cached versus uncached greedy outputs")
        details = []
        msg = entry.get("error", "")
        if msg:
            if "teacher-forced/cached mismatch" in msg:
                errors.append("teacher-forced versus cached output")
            elif "Medusa/cached greedy mismatch" in msg:
                errors.append("static Medusa versus cached output")
            elif "offline/online rank or label mismatch:" in msg:
                details.extend(ast.literal_eval(msg.split("offline/online rank or label mismatch:",1)[1].strip()))
            else:
                errors.append(msg.split(";")[0][:200])
        for policy in entry["policies"]:
            if not policy["exact_agreement"]:
                errors.append("policy output mismatch: "+policy["name"])
            details.extend(policy["rounds"])
        rank=sum(not r["same_ranks"] for r in details)
        prefix=sum(not r["same_prefix"] for r in details)
        labels=sum(len(r["label_mismatches"]) for r in details)
        if rank: errors.append("rank changes")
        if prefix: errors.append("visited prefix mismatch")
        if labels: errors.append("selected node label disagreement")
        slots=[s for r in details for s in r.get("rank_changes",[])]
        rows.append(dict(prompt=entry["prompt"],passed=entry["passed"],findings=errors,
                         policies_completed=len(entry["policies"]),inspected_rounds=len(details),
                         rank_change_rounds=rank,prefix_disagreements=prefix,label_disagreements=labels,
                         recorded_changed_slots=len(slots),changed_selected_slots=sum(s["selected"] for s in slots)))
    output[group]=dict(passed=sum(r["passed"] for r in rows),tested=len(rows),rows=rows)
args.out.write_text(json.dumps(output,indent=2))
