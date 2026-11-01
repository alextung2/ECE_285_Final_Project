"""Budgeted tree selection (proposal Sec. 3.3).

"At each decoding step, expand high-scoring frontier nodes until reaching B
nodes, always retaining ancestors." Since every scorer guarantees
score(child) <= score(parent), best-first expansion returns the top-B nodes by
score and the selected set is automatically ancestor-closed.

The root (the backbone's own token g_0) is not counted in B, matching the
convention that B is the number of *draft* nodes Medusa verifies.
"""

from __future__ import annotations

import heapq
import itertools

import numpy as np

from .traces import TraceSet


def select_tree(scorer, traces: TraceSet, step: int | None, budget: int,
                max_depth: int | None = None, max_rank: int | None = None) -> list[tuple[tuple, float]]:
    """Return exactly budget nodes, or reject an impossible budget.

    step=None builds a step-independent (fixed) tree; only valid for scorers
    with `dynamic = False`.
    """
    if step is None and getattr(scorer, "dynamic", True):
        raise ValueError(f"{scorer.name} is dynamic; a fixed tree needs a static scorer")
    H = traces.n_heads if max_depth is None else min(max_depth, traces.n_heads)
    K = traces.topk if max_rank is None else min(max_rank, traces.topk)
    if not isinstance(budget, (int, np.integer)) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    if H < 1 or K < 1 or budget > sum(K ** d for d in range(1, H + 1)):
        raise ValueError("budget exceeds the candidate tree capacity")
    tie = itertools.count()
    heap: list = []

    def push_children(parent: tuple, pscore: float):
        if len(parent) >= H:
            return
        scores = scorer.child_scores(traces, step, parent, pscore)[:K]
        if len(scores) != K or not np.isfinite(scores).all():
            raise ValueError("scorer must return K finite child scores")
        if np.any(np.asarray(scores) < 0) or np.any(np.asarray(scores) > pscore + 1e-8):
            raise ValueError("child scores must be between zero and the parent score")
        for r, s in enumerate(scores):
            heapq.heappush(heap, (-float(s), next(tie), parent + (r,)))

    push_children((), 1.0)
    tree: list[tuple[tuple, float]] = []
    while heap and len(tree) < budget:
        neg, _, path = heapq.heappop(heap)
        tree.append((path, -neg))
        push_children(path, -neg)
    return tree


def fixed_tree(scorer, traces: TraceSet, budget: int, **kw) -> list[tuple[tuple, float]]:
    """Medusa-style static tree from a step-independent scorer (fit on calibration data)."""
    return select_tree(scorer, traces, None, budget, **kw)


def tree_to_medusa_choices(tree: list[tuple[tuple, float]]) -> list[list[int]]:
    """Convert to Medusa's `medusa_choices` format (list of rank paths, sorted by depth then rank)."""
    return sorted((list(p) for p, _ in tree), key=lambda p: (len(p), p))


def is_ancestor_closed(tree: list[tuple[tuple, float]]) -> bool:
    paths = {p for p, _ in tree}
    return all(p[:-1] in paths or len(p) == 1 for p in paths)


def validate_choices(choices, n_heads=5, topk=10):
    """Validate the rank paths consumed by upstream Medusa buffer construction."""
    paths = [tuple(p) for p in choices]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("tree must be nonempty and contain unique paths")
    for p in paths:
        if not 1 <= len(p) <= n_heads or any(
            not isinstance(r, (int, np.integer)) or not 0 <= r < topk for r in p
        ):
            raise ValueError("invalid tree depth or candidate rank")
    if not is_ancestor_closed([(p, 0.0) for p in paths]):
        raise ValueError("tree must include every ancestor")
    return sorted([list(p) for p in paths], key=lambda p: (len(p), p))
