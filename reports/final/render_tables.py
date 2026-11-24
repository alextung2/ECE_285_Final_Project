"""Render result tables directly from archived study outputs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("study", type=Path, help="Tuned study directory")
parser.add_argument("--out", required=True, type=Path)
parser.add_argument("--offline-only", action="store_true")
args = parser.parse_args()
stud = args.study
def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

names = {"medusa_acc": "Fixed marginal", "count_cond": "Fixed conditional",
         "medusa_plus": "MEDUSA+", "direct": "Direct classifier", "logistic_cond": "Dynamic conditional"}
budgets = [8, 16, 32, 63, 96]
def table(heading, rows):
    lines = [r"\begin{center}", r"\begin{tabular}{l" + "r"*(len(heading)-1) + "}",
             r"\toprule", " & ".join(heading) + r"\\",r"\midrule"]
    lines.extend(" & ".join(row) + r"\\" for row in rows)
    return "\n".join(lines + [r"\bottomrule",r"\end{tabular}",r"\end{center}"])

def signed(value): return f"{float(value):+.3f}"
def interval(row): return f"[{float(row['ci_lo']):.3f}, {float(row['ci_hi']):.3f}]"

comparisons = {(r["method"],int(r["budget"])):r for r in read_csv(stud/"results/tree_comparison.csv")}
assert set(comparisons) == {(m,b) for m in names for b in budgets}
calibration = {r["method"]:r for r in read_csv(stud/"tables/calibration_path.csv") if r["split"].startswith("eval")}
rows = [[label]+[f"{float(comparisons[m,b]['accepted']):.3f}" for b in budgets]+
        [f"{float(calibration[m]['brier']):.5f}",f"{float(calibration[m]['ece']):.5f}"] for m,label in names.items()]
acceptance = table(["Policy","8","16","32","63","96","Brier","ECE"],rows)
differences = []
for b in budgets:
    c,l=comparisons["count_cond",b],comparisons["logistic_cond",b]
    differences.append([str(b),signed(c["delta"]),interval(c),signed(l["delta"]),interval(l)])
deltas=table(["Budget",r"$\Delta$ fixed",r"95\% interval",r"$\Delta$ dynamic",r"95\% interval"],differences)

args.out.mkdir(exist_ok=True, parents=True)
for name, body in [("acceptance", acceptance), ("deltas", deltas)]:
    (args.out/f"{name}.tex").write_text(body+"\n", encoding="utf-8")
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.65), sharex=True, constrained_layout=True)
for ax, m, title, color in zip(axs, ("count_cond", "logistic_cond"),
                              ("Fixed conditional vs. marginal", "Dynamic conditional vs. MEDUSA+"),
                              ("#27645e", "#3e548a")):
    values = [comparisons[m, b] for b in budgets]
    delta = np.array([float(v["delta"]) for v in values])
    lo = np.array([float(v["ci_lo"]) for v in values])
    hi = np.array([float(v["ci_hi"]) for v in values])
    x = np.arange(len(budgets))
    ax.axhline(0, linewidth=0.8, color="#555555")
    ax.errorbar(x, delta, yerr=[delta-lo,hi-delta], fmt="o-", color=color, capsize=3)
    ax.set_xticks(x, budgets)
    ax.set_xlabel("Draft-node budget")
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.spines[["top","right"]].set_visible(False)
axs[0].set_ylabel("Difference in tokens per state", fontsize=9)
limits = [ax.get_ylim() for ax in axs]
for ax in axs:
    ax.set_ylim(min(v[0] for v in limits),max(v[1] for v in limits))
fig.savefig(args.out / "paired_acceptance.pdf")

if args.offline_only:
    raise SystemExit(0)

timing = {b:{r["method"]:r for r in read_csv(stud/f"results/speed_B{b}.csv")} for b in (16,63)}
gpu_rows = []
for m,label in {"greedy":"Cached greedy",**names}.items():
    cells=[label]
    for b in (16,63):
        r=timing[b][m]
        cells.extend([f"{float(r['tokens_per_s']):.2f}",f"{float(r['speedup']):.3f}",f"{round(float(r['agree'])*8)}/8"])
    gpu_rows.append(cells)
gpu = table(["Policy",r"tok/s$_{16}$",r"ratio$_{16}$","exact",r"tok/s$_{63}$",r"ratio$_{63}$","exact"],gpu_rows)

panel = json.loads((stud/"results/cached_verification.json").read_text())
assert len(panel) == 8*5*2
rows = [r for p in panel for r in p["rounds"]]
summary = dict(policy_prompt_runs=len(panel), outputs_agree=sum(p["output_agrees"] for p in panel),
               saved_reference_agrees=sum(p["saved_reference_agrees"] for p in panel),
               complete_rounds=len(rows), divergent_prefix=sum(not r["same_prefix"] for r in rows),
               rank_changed_rounds=sum(not r["same_ranks"] for r in rows),
               changed_selected_slots=sum(sum(x["selected"] for x in r["rank_changes"]) for r in rows),
               changed_rank_slots=sum(len(r["rank_changes"]) for r in rows),
               selected_nodes=sum(r["nodes_checked"] for r in rows),
               label_mismatches=sum(len(r["label_mismatches"]) for r in rows))
(args.out/"timing.tex").write_text(gpu+"\n", encoding="utf-8")
(args.out/"agreement_summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
