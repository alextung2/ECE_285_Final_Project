"""Node batches and features x available *before* verification.

A NodeBatch is a flat list of tree nodes across many decoding steps. Every
learned estimator (logistic conditional, direct classifier) uses the SAME
feature map, so the comparison in the proposal is feature- and
capacity-matched by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .traces import TraceSet

EPS = 1e-6


@dataclass
class NodeBatch:
    step: np.ndarray    # [M] index into TraceSet
    path: np.ndarray    # [M, H] rank path, padded with -1
    depth: np.ndarray   # [M] in 1..H

    def __len__(self) -> int:
        return len(self.step)

    @property
    def own_rank(self) -> np.ndarray:
        return self.path[np.arange(len(self)), self.depth - 1]

    @property
    def parent_rank(self) -> np.ndarray:
        """-1 for depth-1 nodes (their parent is the always-accepted root g_0)."""
        pr = np.full(len(self), -1, dtype=self.path.dtype)
        deep = self.depth > 1
        pr[deep] = self.path[np.where(deep)[0], self.depth[deep] - 2]
        return pr

    @staticmethod
    def concat(parts: list["NodeBatch"]) -> "NodeBatch":
        return NodeBatch(np.concatenate([p.step for p in parts]),
                         np.concatenate([p.path for p in parts]),
                         np.concatenate([p.depth for p in parts]))


# ---------------------------------------------------------------------------
# Training-set builders
# ---------------------------------------------------------------------------

def conditional_nodes(traces: TraceSet, max_rank: int | None = None) -> tuple[NodeBatch, np.ndarray]:
    """All children of *accepted* parents, with label S_v given S_parent = 1.

    "Conditional labels use only calibration examples whose parent prefix was
    accepted" (proposal, Sec. 3.2). Under greedy verification the accepted
    parent at depth d-1 is unique (the gold prefix), so each step contributes
    at most H * K examples.
    """
    K = traces.topk if max_rank is None else min(max_rank, traces.topk)
    H = traces.n_heads
    L = traces.accept_chain_len()               # gold prefix of length L is in candidate space
    steps, paths, depths = [], [], []
    for d in range(1, H + 1):
        # parent (depth d-1) accepted <=> gold prefix of length d-1 is representable
        s = np.where(L >= d - 1)[0]
        if len(s) == 0:
            continue
        p = np.full((len(s), K, H), -1, dtype=np.int16)
        p[:, :, : d - 1] = traces.gold_rank[s, None, : d - 1]
        p[:, :, d - 1] = np.arange(K)[None, :]
        steps.append(np.repeat(s, K))
        paths.append(p.reshape(-1, H))
        depths.append(np.full(len(s) * K, d, dtype=np.int16))
    nb = NodeBatch(np.concatenate(steps), np.concatenate(paths), np.concatenate(depths))
    y = traces.is_accepted(nb.step, nb.path, nb.depth)
    return nb, y


def candidate_pool_nodes(traces: TraceSet, scorer, budget: int, n_steps: int | None = None,
                         seed: int = 0) -> tuple[NodeBatch, np.ndarray]:
    """Nodes from a reference tree per step (default: MEDUSA+ prob product).

    Used to (i) train the direct classifier, whose label is full-prefix
    acceptance S_v on nodes whose parents may have failed, and (ii) compute
    path-level Brier scores on the *same* candidate set for every method.
    """
    from .tree import select_tree  # local import to avoid cycle

    rng = np.random.default_rng(seed)
    idx = np.arange(traces.n_steps)
    if n_steps is not None and n_steps < len(idx):
        idx = rng.choice(idx, n_steps, replace=False)
    H = traces.n_heads
    steps, paths, depths = [], [], []
    for s in idx:
        for path, _ in select_tree(scorer, traces, int(s), budget):
            row = np.full(H, -1, dtype=np.int16)
            row[: len(path)] = path
            steps.append(s), paths.append(row), depths.append(len(path))
    nb = NodeBatch(np.array(steps), np.stack(paths), np.array(depths, dtype=np.int16))
    return nb, traces.is_accepted(nb.step, nb.path, nb.depth)


# ---------------------------------------------------------------------------
# Feature map (shared by logistic-conditional and direct classifier)
# ---------------------------------------------------------------------------

def _entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, EPS, 1.0)
    return -(q * np.log(q)).sum(-1)


def node_features(traces: TraceSet, nb: NodeBatch, rank_cap: int = 4) -> np.ndarray:
    """Features from the proposal: ranks, confidence, depth, parent-child interactions.

    All features are computable before verification (no gold info).
    Returns float32 [M, F].
    """
    H = traces.n_heads
    M = len(nb)
    d0 = nb.depth.astype(np.int64) - 1                 # 0-based head index
    r = nb.own_rank.astype(np.int64)
    pr = nb.parent_rank.astype(np.int64)
    has_parent = (nb.depth > 1).astype(np.float32)

    probs = traces.topk_prob[nb.step]                  # [M, H, K]
    own_p = probs[np.arange(M), d0, r]
    par_p = np.where(has_parent > 0, probs[np.arange(M), np.maximum(d0 - 1, 0), np.maximum(pr, 0)],
                     traces.root_prob[nb.step])
    head_top1 = probs[np.arange(M), d0, 0]
    head_ent = _entropy(probs[np.arange(M), d0])

    lp, lpp = np.log(own_p + EPS), np.log(par_p + EPS)
    # log of the MEDUSA+ path score (sum of head log-probs along the path). C2T uses
    # this kind of joint-probability feature; it's shared by both learned models.
    on_path = np.arange(H)[None, :] < nb.depth[:, None]
    pp = probs[np.arange(M)[:, None], np.arange(H)[None, :], np.maximum(nb.path, 0)]
    log_path_p = np.where(on_path, np.log(pp + EPS), 0.0).sum(1)

    def onehot(v, n):
        out = np.zeros((M, n), dtype=np.float32)
        out[np.arange(M), np.clip(v, 0, n - 1)] = 1
        return out

    feats = [
        onehot(d0, H),                                 # depth
        onehot(r, rank_cap + 1),                       # own rank (capped)
        onehot(np.where(has_parent > 0, pr, rank_cap + 1), rank_cap + 2),  # parent rank (+ "root")
        lp[:, None], lpp[:, None],                     # confidences
        log_path_p[:, None],                           # joint (product) confidence of the path
        (own_p / (head_top1 + EPS))[:, None],          # relative confidence within head
        head_ent[:, None],                             # head uncertainty
        np.log(traces.root_prob[nb.step] + EPS)[:, None],
        (lp * lpp)[:, None],                           # parent-child interaction
        ((r == 0) & (pr <= 0)).astype(np.float32)[:, None],  # "both top-1" interaction
        (lp * d0)[:, None],                            # confidence x depth
    ]
    return np.concatenate(feats, axis=1).astype(np.float32)
