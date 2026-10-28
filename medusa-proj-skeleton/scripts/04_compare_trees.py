"""Weeks 4-5: matched-budget tree comparisons on eval traces (trace-simulated).

  Fixed trees:    original Medusa (accuracy product)   vs  count-based conditional
  Dynamic trees:  MEDUSA+ (prob product)  vs  direct classifier (C2T-inspired)  vs  ours

Same candidates (top-K per head), same best-first selection, same budget B.
Primary metric: mean accepted tokens per verification step (incl. the backbone token).
Differences vs. the relevant baseline come with prompt-level bootstrap 95% CIs.

These are TRACE-ONLY numbers: never report them as measured speedups.
"""

import numpy as np
from _common import parse, write_table


def main():
    args, cfg = parse(__doc__)
    from mcal.config import out_path
    from mcal.estimators import Scorer
    from mcal.evaluate import evaluate_tree_policy, paired_bootstrap
    from mcal.traces import TraceSet
    from mcal.tree import fixed_tree

    calib = TraceSet.load(out_path(cfg, cfg["traces"]["calib"]["file"]))
    fit, _ = calib.split_by_prompt(cfg["estimators"]["val_frac"], seed=cfg["seed"])
    ev = TraceSet.load(out_path(cfg, cfg["traces"]["eval"]["file"]))
    S = {n: Scorer.load(out_path(cfg, "models", f"{n}.pkl"))
         for n in ["medusa_acc", "medusa_plus", "count_cond", "logistic_cond", "direct"]}

    rng = np.random.default_rng(cfg["seed"])
    n_eval = cfg["compare"].get("eval_steps")
    steps = np.sort(rng.choice(ev.n_steps, n_eval, replace=False)) if n_eval and n_eval < ev.n_steps \
        else np.arange(ev.n_steps)
    groups = ev.prompt_id[steps]

    # (method, kind, baseline-for-CI)
    plan = [("medusa_acc", "fixed", None), ("count_cond", "fixed", "medusa_acc"),
            ("medusa_plus", "dynamic", None), ("direct", "dynamic", "medusa_plus"),
            ("logistic_cond", "dynamic", "medusa_plus")]

    rows, curves = [], {}
    for B in cfg["compare"]["budgets"]:
        res = {}
        for name, kind, _ in plan:
            ft = fixed_tree(S[name], fit, B) if kind == "fixed" else None
            res[name] = evaluate_tree_policy(S[name], ev, B, fixed=ft, steps=steps)
        for name, kind, base in plan:
            r = res[name]
            row = dict(budget=B, tree=kind, method=name, accepted=r["mean_accepted"],
                       sem=r["sem_accepted"], predicted=r["mean_predicted"],
                       build_ms=r["tree_build_ms"])
            if base:
                d, lo, hi = paired_bootstrap(r["per_step_accepted"], res[base]["per_step_accepted"], groups)
                row.update(delta_vs=base, delta=d, ci_lo=lo, ci_hi=hi)
            rows.append(row)
            curves.setdefault(name, []).append(r["mean_accepted"])
        print(f"B={B}: " + ", ".join(f"{n}={res[n]['mean_accepted']:.3f}" for n, _, _ in plan))

    cols = ["budget", "tree", "method", "accepted", "sem", "predicted", "delta_vs", "delta",
            "ci_lo", "ci_hi", "build_ms"]
    print(write_table(rows, out_path(cfg, "results", "tree_comparison.csv"), cols))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, kind, _ in plan:
        ax.plot(cfg["compare"]["budgets"], curves[name], "o-" if kind == "dynamic" else "s--", label=name)
    ax.set_xlabel("tree budget B (draft nodes)"), ax.set_ylabel("accepted tokens / step (trace-sim.)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path(cfg, "results", "accepted_vs_budget.png"), dpi=150)


if __name__ == "__main__":
    main()
