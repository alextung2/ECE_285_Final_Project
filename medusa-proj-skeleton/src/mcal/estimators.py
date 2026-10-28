"""Path scorers compared in the proposal.

Every scorer exposes the same two methods so tree selection and evaluation
are identical across methods (matched candidates, procedure, budget):

    child_scores(traces, step, parent_path, parent_score) -> [K] path scores
    path_scores(traces, nodes)                            -> [M] path scores

Edge-factor scorers define factor(traces, nodes) = estimate for the LAST edge
of each node and score a path by the product of its factors (Eq. 1):

    P_hat(S_v | x) = prod_{e in path(v)} q_hat_e(x)

  name               factor q_e                                  tree
  -----------------  ------------------------------------------  ---------
  MedusaAccuracy     marginal acc of (head, rank)                fixed     (original Medusa)
  MedusaPlusProb     head softmax prob p_{j, r_j}                dynamic   (MEDUSA+)
  CountConditional   smoothed P(S_v | S_u, head, rank, parent rank)  fixed (ours, step 1)
  LogisticConditional  logistic P(S_v | S_u, x)                  dynamic   (ours)
  DirectClassifier   logistic P(S_v | x) on the FULL prefix      dynamic   (C2T-inspired baseline)

Because every q_e is in [0, 1], child scores never exceed parent scores, which
is what lets best-first selection keep the tree ancestor-closed.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .features import NodeBatch, conditional_nodes, node_features
from .traces import TraceSet


class Scorer:
    name = "base"
    dynamic = True   # False => scores don't depend on the step (fixed tree)

    # -- to implement ----------------------------------------------------
    def fit(self, traces: TraceSet, **kw) -> "Scorer":
        return self

    def factor(self, traces: TraceSet, nodes: NodeBatch) -> np.ndarray:
        raise NotImplementedError

    # -- shared ----------------------------------------------------------
    def child_scores(self, traces: TraceSet, step: int | None, parent_path: tuple,
                     parent_score: float) -> np.ndarray:
        nodes = _children(traces, step, parent_path)
        return parent_score * self.factor(traces, nodes)

    def path_scores(self, traces: TraceSet, nodes: NodeBatch) -> np.ndarray:
        score = np.ones(len(nodes))
        for d in range(1, int(nodes.depth.max()) + 1):
            m = nodes.depth >= d
            pref = NodeBatch(nodes.step[m], _truncate(nodes.path[m], d), np.full(m.sum(), d, dtype=np.int16))
            score[m] *= self.factor(traces, pref)
        return score

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "Scorer":
        with open(path, "rb") as f:
            return pickle.load(f)


def _truncate(path: np.ndarray, d: int) -> np.ndarray:
    out = path.copy()
    out[:, d:] = -1
    return out


def _children(traces: TraceSet, step: int | None, parent_path: tuple) -> NodeBatch:
    H, K, d = traces.n_heads, traces.topk, len(parent_path) + 1
    path = np.full((K, H), -1, dtype=np.int16)
    path[:, : d - 1] = parent_path
    path[:, d - 1] = np.arange(K)
    s = np.zeros(K, dtype=np.int64) if step is None else np.full(K, step, dtype=np.int64)
    return NodeBatch(s, path, np.full(K, d, dtype=np.int16))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class MedusaAccuracy(Scorer):
    """Original Medusa: product of per-(head, rank) marginal top-k accuracies."""
    name, dynamic = "medusa_acc", False

    def fit(self, traces, **kw):
        H, K = traces.n_heads, traces.topk
        self.acc = np.stack([[(traces.gold_rank[:, j] == r).mean() for r in range(K)] for j in range(H)])
        return self

    def factor(self, traces, nodes):
        return self.acc[nodes.depth - 1, nodes.own_rank]


class MedusaPlusProb(Scorer):
    """MEDUSA+: product of head softmax probabilities (no fitting)."""
    name, dynamic = "medusa_plus", True

    def factor(self, traces, nodes):
        return traces.topk_prob[nodes.step, nodes.depth - 1, nodes.own_rank].astype(np.float64)


# ---------------------------------------------------------------------------
# Ours
# ---------------------------------------------------------------------------

class CountConditional(Scorer):
    """Smoothed count table q[head, parent_rank, rank] = P(S_v | S_u = 1).

    Indexed by head and preceding-token rank (proposal Sec. 3.2). Smoothing
    shrinks toward the marginal-accuracy factor with strength `alpha`, so
    rare cells fall back to the original Medusa estimate.
    Parent rank index K is reserved for "parent = root" (depth-1 nodes).
    """
    name, dynamic = "count_cond", False

    def __init__(self, alpha: float = 20.0):
        self.alpha = alpha

    def fit(self, traces, **kw):
        H, K = traces.n_heads, traces.topk
        nb, y = conditional_nodes(traces)
        pr = np.where(nb.depth > 1, nb.parent_rank, K)
        d0, r = nb.depth - 1, nb.own_rank
        hits = np.zeros((H, K + 1, K))
        tot = np.zeros((H, K + 1, K))
        np.add.at(tot, (d0, pr, r), 1)
        np.add.at(hits, (d0, pr, r), y)
        # prior: P(S_v | S_u) pooled over parent ranks
        prior = hits.sum(1) / np.maximum(tot.sum(1), 1)            # [H, K]
        self.q = (hits + self.alpha * prior[:, None, :]) / (tot + self.alpha)
        self.counts = tot
        return self

    def factor(self, traces, nodes):
        K = traces.topk
        pr = np.where(nodes.depth > 1, nodes.parent_rank, K)
        return self.q[nodes.depth - 1, pr, nodes.own_rank]


class _Logistic:
    """Tiny numpy-side wrapper so prediction inside the tree loop is fast."""

    def __init__(self, C: float = 1.0, max_iter: int = 2000):
        self.C, self.max_iter = C, max_iter

    def fit(self, X, y, sample_weight=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler().fit(X)
        clf = LogisticRegression(C=self.C, max_iter=self.max_iter)
        clf.fit(self.scaler.transform(X), y.astype(int), sample_weight=sample_weight)
        self.w, self.b = clf.coef_[0], clf.intercept_[0]
        return self

    def predict_proba(self, X):
        z = ((X - self.scaler.mean_) / self.scaler.scale_) @ self.w + self.b
        return 1 / (1 + np.exp(-z))


class LogisticConditional(Scorer):
    """Logistic regression for q_v(x) = P(S_v | S_u = 1, x), trained on accepted-parent nodes."""
    name, dynamic = "logistic_cond", True

    def __init__(self, C: float = 1.0, rank_cap: int = 4):
        self.C, self.rank_cap = C, rank_cap

    def fit(self, traces, **kw):
        nb, y = conditional_nodes(traces)
        self.model = _Logistic(self.C).fit(node_features(traces, nb, self.rank_cap), y)
        return self

    def factor(self, traces, nodes):
        return self.model.predict_proba(node_features(traces, nodes, self.rank_cap))


class DirectClassifier(Scorer):
    """C2T-inspired baseline: one classifier predicts full-prefix acceptance P(S_v | x).

    Same features and capacity (one logistic layer) as LogisticConditional;
    the only difference is the *target*: unconditional S_v on a candidate pool
    rather than S_v | S_u = 1. With monotone=True a child's score is capped at
    its parent's so best-first selection stays ancestor-closed (document this
    choice in the report; C2T itself prunes by threshold).
    """
    name, dynamic = "direct", True

    def __init__(self, C: float = 1.0, rank_cap: int = 4, monotone: bool = True):
        self.C, self.rank_cap, self.monotone = C, rank_cap, monotone

    def fit(self, traces, pool: tuple[NodeBatch, np.ndarray] | None = None, **kw):
        if pool is None:
            raise ValueError("DirectClassifier needs a candidate pool: see features.candidate_pool_nodes")
        nb, y = pool
        self.model = _Logistic(self.C).fit(node_features(traces, nb, self.rank_cap), y)
        return self

    def _raw(self, traces, nodes):
        return self.model.predict_proba(node_features(traces, nodes, self.rank_cap))

    def child_scores(self, traces, step, parent_path, parent_score):
        s = self._raw(traces, _children(traces, step, parent_path))
        return np.minimum(s, parent_score) if self.monotone else s

    def path_scores(self, traces, nodes):
        s = self._raw(traces, nodes)
        if not self.monotone:
            return s
        for d in range(1, int(nodes.depth.max())):   # cap by every ancestor
            m = nodes.depth > d
            pref = NodeBatch(nodes.step[m], _truncate(nodes.path[m], d), np.full(m.sum(), d, dtype=np.int16))
            s[m] = np.minimum(s[m], self._raw(traces, pref))
        return s


REGISTRY = {c.name: c for c in [MedusaAccuracy, MedusaPlusProb, CountConditional,
                                LogisticConditional, DirectClassifier]}
