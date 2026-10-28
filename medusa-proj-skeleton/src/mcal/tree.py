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
    """Return [(rank_path, path_score), ...] of length <= budget, in selection order.

    step=None builds a step-independent (fixed) tree; only valid for scorers
    with `dynamic = False`.
    """
    if step is None and getattr(scorer, "dynamic", True):
        raise ValueError(f"{scorer.name} is dynamic; a fixed tree needs a static scorer")
    H = traces.n_heads if max_depth is None else min(max_depth, traces.n_heads)
    K = traces.topk if max_rank is None else min(max_rank, traces.topk)
    tie = itertools.count()
    heap: list = []

    def push_children(parent: tuple, pscore: float):
        if len(parent) >= H:
            return
        scores = scorer.child_scores(traces, step, parent, pscore)[:K]
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
