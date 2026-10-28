"""Step 1 of the method: measure product-score error and its direction.

Example from the proposal: head accuracies 0.80 and 0.75 give a product of
0.60; an observed joint success rate of 0.70 means the product UNDERestimates.
For each short rank path we compare
    product  = prod_j P(gold_rank[j] == r_j)          (original Medusa)
    observed = P(gold_rank[1..d] == r_1..r_d)         (true joint acceptance)
"""

from __future__ import annotations

import itertools

import numpy as np

from .traces import TraceSet


def product_vs_joint(traces: TraceSet, max_depth: int = 3, max_rank: int = 3) -> list[dict]:
    g = traces.gold_rank
    H = min(max_depth, traces.n_heads)
    marg = [[(g[:, j] == r).mean() for r in range(traces.topk)] for j in range(traces.n_heads)]
    rows = []
    for d in range(1, H + 1):
        for path in itertools.product(range(max_rank), repeat=d):
            prod = float(np.prod([marg[j][r] for j, r in enumerate(path)]))
            hit = np.all(g[:, :d] == np.array(path)[None, :], axis=1)
            obs = float(hit.mean())
            n = int(hit.sum())
            se = np.sqrt(max(obs * (1 - obs), 1e-12) / len(g))
            rows.append(dict(path=str(list(path)), depth=d, product=prod, observed=obs,
                             ratio=obs / prod if prod > 0 else np.nan,
                             error=prod - obs,
                             direction="under" if prod < obs - 2 * se else
                                       "over" if prod > obs + 2 * se else "ok",
                             n_hits=n))
    return rows


def conditional_vs_marginal(traces: TraceSet) -> list[dict]:
    """P(head j top-1 correct | head j-1 top-1 correct) vs P(head j top-1 correct)."""
    g = traces.gold_rank
    rows = []
    for j in range(1, traces.n_heads):
        prev = np.all(g[:, :j] == 0, axis=1)
        cur = g[:, j] == 0
        rows.append(dict(head=j + 1, marginal=float(cur.mean()),
                         conditional=float(cur[prev].mean()) if prev.any() else np.nan,
                         n_parent=int(prev.sum())))
    return rows
