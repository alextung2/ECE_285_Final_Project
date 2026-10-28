"""Trace-simulated evaluation.

Primary metric (proposal Sec. 5): mean accepted tokens per verification step
at matched budget B. On traces this is exact for greedy verification:

    accepted(T) = 1 + |{v in T : S_v = 1}|     (the 1 is the backbone's g_0)

and a calibrated scorer predicts it as 1 + sum_{v in T} P_hat(S_v | x).

IMPORTANT: numbers from this module are *trace-only*. Per the proposal they
must not be reported as speedups; wall-clock tokens/sec comes from decode.py.
"""

from __future__ import annotations

import time

import numpy as np

from .features import NodeBatch
from .traces import TraceSet
from .tree import select_tree


def accepted_tokens(traces: TraceSet, step: int, tree_paths: set[tuple]) -> int:
    gold = traces.gold_rank[step]
    n = 0
    for d in range(1, traces.n_heads + 1):
        if gold[d - 1] < 0 or tuple(int(g) for g in gold[:d]) not in tree_paths:
            break
        n += 1
    return 1 + n


def evaluate_tree_policy(scorer, traces: TraceSet, budget: int, fixed=None,
                         steps: np.ndarray | None = None) -> dict:
    """Evaluate a scorer with dynamic per-step trees, or a given fixed tree.

    fixed: output of tree.fixed_tree(...) to evaluate a static tree instead.
    """
    steps = np.arange(traces.n_steps) if steps is None else steps
    acc, pred, build_t = np.zeros(len(steps)), np.zeros(len(steps)), np.zeros(len(steps))
    if fixed is not None:
        fixed_set = {p for p, _ in fixed}
    for i, s in enumerate(steps):
        s = int(s)
        if fixed is None:
            t0 = time.perf_counter()
            tree = select_tree(scorer, traces, s, budget)
            build_t[i] = time.perf_counter() - t0
            paths = {p for p, _ in tree}
            pred[i] = 1 + sum(sc for _, sc in tree)
        else:
            paths = fixed_set
            pred[i] = 1 + sum(sc for _, sc in fixed)
        acc[i] = accepted_tokens(traces, s, paths)
    return dict(
        method=scorer.name if fixed is None else f"{scorer.name}[fixed]",
        budget=budget,
        mean_accepted=float(acc.mean()),
        sem_accepted=float(acc.std(ddof=1) / np.sqrt(len(acc))),
        mean_predicted=float(pred.mean()),
        tree_build_ms=float(build_t.mean() * 1e3),  # Python-side; NOT representative of a GPU impl
        n_steps=len(steps),
        per_step_accepted=acc,
    )


def paired_bootstrap(a: np.ndarray, b: np.ndarray, groups: np.ndarray, n_boot: int = 2000,
                     seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) with a 95% CI, resampling *prompts* (steps within a prompt are correlated)."""
    rng = np.random.default_rng(seed)
    diff = a - b
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=diff, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(uniq), len(uniq))
        boots[i] = sums[idx].sum() / cnts[idx].sum()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y.astype(float)) ** 2))


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    out = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(out)


def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean predicted, observed rate, count) per bin, for reliability diagrams."""
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    mp, obs, cnt = [], [], []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            mp.append(p[m].mean()), obs.append(y[m].mean()), cnt.append(m.sum())
    return np.array(mp), np.array(obs), np.array(cnt)


def calibration_report(scorer, traces: TraceSet, pool: tuple[NodeBatch, np.ndarray]) -> dict:
    """Path-level calibration of P_hat(S_v | x) on a shared candidate pool."""
    nb, y = pool
    p = np.clip(scorer.path_scores(traces, nb), 0, 1)
    return dict(method=scorer.name, brier=brier(p, y), ece=ece(p, y),
                mean_pred=float(p.mean()), mean_obs=float(y.mean()))
