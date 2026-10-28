"""Week 2: diagnose product-score error (proposal Sec. 3, step 1).

Outputs (in <out_dir>/diagnose/):
  product_vs_joint.csv/.md   observed joint acceptance vs product of head accuracies per rank path
  conditional_vs_marginal.*  P(head j top-1 | previous heads top-1) vs P(head j top-1)
  product_vs_joint.png       scatter, points above the diagonal = product UNDERestimates
"""

import numpy as np
from _common import parse, write_table


def main():
    args, cfg = parse(__doc__)
    from mcal.config import out_path
    from mcal.diagnose import conditional_vs_marginal, product_vs_joint
    from mcal.traces import TraceSet

    ts = TraceSet.load(out_path(cfg, cfg["traces"]["calib"]["file"]))
    print(ts.summary())

    rows = product_vs_joint(ts, max_depth=3, max_rank=3)
    md = write_table(rows, out_path(cfg, "diagnose", "product_vs_joint.csv"))
    print(md)
    dirs = [r["direction"] for r in rows]
    print({d: dirs.count(d) for d in set(dirs)})

    rows2 = conditional_vs_marginal(ts)
    print(write_table(rows2, out_path(cfg, "diagnose", "conditional_vs_marginal.csv")))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for d, m in zip([1, 2, 3], ["o", "s", "^"]):
        sel = [r for r in rows if r["depth"] == d]
        ax.scatter([r["product"] for r in sel], [r["observed"] for r in sel], marker=m, label=f"depth {d}")
    lim = max(max(r["product"] for r in rows), max(r["observed"] for r in rows)) * 1.05
    ax.plot([1e-4, lim], [1e-4, lim], "k--", lw=1)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("product of head accuracies"), ax.set_ylabel("observed joint acceptance")
    ax.set_title("Above diagonal = product underestimates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path(cfg, "diagnose", "product_vs_joint.png"), dpi=150)
    print("saved figures to", out_path(cfg, "diagnose", "x").parent)


if __name__ == "__main__":
    main()
