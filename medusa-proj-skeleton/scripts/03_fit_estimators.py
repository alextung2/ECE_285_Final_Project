"""Week 3: fit count, direct and conditional estimators on calibration traces.

Calibration traces are split by prompt into fit / held-out parts; calibration
quality (Brier, ECE) is reported on the held-out part AND on the eval traces,
on one shared candidate pool so every method is scored on the same nodes.

Outputs (in <out_dir>/models/ and <out_dir>/results/):
  <name>.pkl                         fitted scorers
  fixed_tree_<name>_B<b>.json        fixed trees in Medusa's medusa_choices format
  calibration.csv/.md, reliability.png
"""

import json

import numpy as np
from _common import parse, write_table


def main():
    args, cfg = parse(__doc__)
    from mcal.config import out_path
    from mcal.estimators import (CountConditional, DirectClassifier, LogisticConditional,
                                 MedusaAccuracy, MedusaPlusProb)
    from mcal.evaluate import calibration_report, reliability
    from mcal.features import candidate_pool_nodes
    from mcal.traces import TraceSet
    from mcal.tree import fixed_tree, tree_to_medusa_choices

    ecfg = cfg["estimators"]
    calib = TraceSet.load(out_path(cfg, cfg["traces"]["calib"]["file"]))
    evals = TraceSet.load(out_path(cfg, cfg["traces"]["eval"]["file"]))
    fit, held = calib.split_by_prompt(ecfg["val_frac"], seed=cfg["seed"])
    print(f"fit: {fit.n_steps} steps | held-out calib: {held.n_steps} | eval: {evals.n_steps}")

    ref = MedusaPlusProb()
    pc = ecfg["pool"]
    pool_fit = candidate_pool_nodes(fit, ref, pc["budget"], pc["n_steps"], seed=cfg["seed"])
    print(f"direct-classifier pool: {len(pool_fit[0])} nodes, positive rate {pool_fit[1].mean():.3f}")

    scorers = [
        MedusaAccuracy().fit(fit),
        ref,
        CountConditional(**ecfg["count_cond"]).fit(fit),
        LogisticConditional(**ecfg["logistic_cond"]).fit(fit),
        DirectClassifier(**ecfg["direct"]).fit(fit, pool=pool_fit),
    ]
    for s in scorers:
        s.save(out_path(cfg, "models", f"{s.name}.pkl"))

    # fixed trees (static scorers only), in medusa_choices format for online decoding
    for s in scorers:
        if not s.dynamic:
            for B in cfg["compare"]["budgets"]:
                ch = tree_to_medusa_choices(fixed_tree(s, fit, B))
                out_path(cfg, "models", f"fixed_tree_{s.name}_B{B}.json").write_text(json.dumps(ch))

    # calibration on shared pools
    rows, curves = [], {}
    for split_name, ts, seed in [("calib_heldout", held, 1), ("eval", evals, 2)]:
        pool = candidate_pool_nodes(ts, ref, pc["budget"], pc["n_steps"], seed=seed)
        for s in scorers:
            r = calibration_report(s, ts, pool)
            r["split"] = split_name
            rows.append(r)
            if split_name == "eval":
                p = np.clip(s.path_scores(ts, pool[0]), 0, 1)
                curves[s.name] = reliability(p, pool[1].astype(float))
    print(write_table(rows, out_path(cfg, "results", "calibration.csv"),
                      ["split", "method", "brier", "ece", "mean_pred", "mean_obs"]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for name, (mp, obs, _) in curves.items():
        ax.plot(mp, obs, "o-", ms=3, label=name)
    ax.set_xlabel("predicted P(S_v)"), ax.set_ylabel("observed acceptance")
    ax.set_title("Path-level reliability (eval pool)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path(cfg, "results", "reliability.png"), dpi=150)


if __name__ == "__main__":
    main()
