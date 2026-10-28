"""Verification traces: the single data format every estimator and evaluation uses.

Notation (matches the proposal)
-------------------------------
At a decoding step the backbone has just produced token g_0 (always accepted
under greedy decoding). Medusa head j (j = 1..H) proposes a top-K list for the
token at offset j. A tree node v at depth d is a *rank path* (r_1, ..., r_d):
"take head 1's rank-r_1 token, then head 2's rank-r_2 token, ...".

Under greedy verification, node v is accepted (S_v = 1) iff its whole prefix
matches the backbone's greedy continuation g_1..g_d, i.e.

    S_v = 1  <=>  gold_rank[j] == r_j  for all j <= d

where gold_rank[j] is the rank of g_j in head j's top-K list (-1 if absent).
So a step is fully described by `gold_rank` (labels) plus the head
probabilities (features). Because only one path can match, the number of
accepted draft tokens for a tree T is simply |{v in T : S_v = 1}|.

Why offline traces are valid: Medusa heads read only the hidden state of the
last accepted token, so running ordinary greedy generation and applying the
heads at *every* position gives exactly the drafts Medusa would see at that
position. (Positions visited during real Medusa decoding are a subset; check
that this doesn't bias results by comparing against online runs in Week 6.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class TraceSet:
    gold_rank: np.ndarray   # [N, H] int16, rank of greedy token g_j in head j's top-K, -1 if absent
    topk_prob: np.ndarray   # [N, H, K] float32, head j's softmax prob at each rank (sorted desc)
    root_prob: np.ndarray   # [N] float32, backbone's prob of g_0 (confidence feature)
    prompt_id: np.ndarray   # [N] int32, which prompt the step came from (for splits)
    position: np.ndarray    # [N] int32, generated-token index within the response
    meta: dict = field(default_factory=dict)
    topk_token: np.ndarray | None = None  # [N, H, K] int32, optional (debugging only)

    # ---- shape helpers -------------------------------------------------
    @property
    def n_steps(self) -> int:
        return self.gold_rank.shape[0]

    @property
    def n_heads(self) -> int:
        return self.gold_rank.shape[1]

    @property
    def topk(self) -> int:
        return self.topk_prob.shape[2]

    # ---- label helpers -------------------------------------------------
    def accept_chain_len(self) -> np.ndarray:
        """[N] longest d such that the gold path's first d ranks are all in top-K.

        This is the acceptance length of an *infinite* tree; any finite tree
        accepts at most this many draft tokens.
        """
        ok = self.gold_rank >= 0
        # first False along heads (H if none)
        first_bad = np.where(ok.all(1), self.n_heads, np.argmin(ok, axis=1))
        return first_bad.astype(np.int32)

    def is_accepted(self, steps: np.ndarray, paths: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Vectorised S_v for nodes (steps[i], paths[i, :depth[i]]). paths padded with -1."""
        gold = self.gold_rank[steps]                       # [M, H]
        match = gold == paths                              # [M, H]
        mask = np.arange(self.n_heads)[None, :] < depth[:, None]
        return np.all(match | ~mask, axis=1)

    # ---- subsetting / splitting -----------------------------------------
    def subset(self, idx: np.ndarray) -> "TraceSet":
        return TraceSet(
            gold_rank=self.gold_rank[idx],
            topk_prob=self.topk_prob[idx],
            root_prob=self.root_prob[idx],
            prompt_id=self.prompt_id[idx],
            position=self.position[idx],
            meta=dict(self.meta),
            topk_token=None if self.topk_token is None else self.topk_token[idx],
        )

    def split_by_prompt(self, frac: float, seed: int = 0) -> tuple["TraceSet", "TraceSet"]:
        """Split by prompt (never by step) so correlated steps don't leak across splits."""
        rng = np.random.default_rng(seed)
        prompts = np.unique(self.prompt_id)
        rng.shuffle(prompts)
        n_a = int(round(frac * len(prompts)))
        a = np.isin(self.prompt_id, prompts[:n_a])
        return self.subset(np.where(a)[0]), self.subset(np.where(~a)[0])

    # ---- I/O ------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(
            gold_rank=self.gold_rank, topk_prob=self.topk_prob, root_prob=self.root_prob,
            prompt_id=self.prompt_id, position=self.position,
            meta=np.array(json.dumps(self.meta)),
        )
        if self.topk_token is not None:
            arrays["topk_token"] = self.topk_token
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "TraceSet":
        z = np.load(path, allow_pickle=False)
        return cls(
            gold_rank=z["gold_rank"], topk_prob=z["topk_prob"], root_prob=z["root_prob"],
            prompt_id=z["prompt_id"], position=z["position"],
            meta=json.loads(str(z["meta"])),
            topk_token=z["topk_token"] if "topk_token" in z.files else None,
        )

    @classmethod
    def concat(cls, parts: list["TraceSet"]) -> "TraceSet":
        cat = lambda name: np.concatenate([getattr(p, name) for p in parts])
        tok = None if any(p.topk_token is None for p in parts) else cat("topk_token")
        return cls(cat("gold_rank"), cat("topk_prob"), cat("root_prob"),
                   cat("prompt_id"), cat("position"), dict(parts[0].meta), tok)

    def summary(self) -> str:
        L = self.accept_chain_len()
        top1 = [(self.gold_rank[:, j] == 0).mean() for j in range(self.n_heads)]
        return (f"TraceSet: {self.n_steps} steps, {len(np.unique(self.prompt_id))} prompts, "
                f"H={self.n_heads}, K={self.topk}\n"
                f"  head top-1 acc: {np.round(top1, 3).tolist()}\n"
                f"  mean oracle chain length (top-K, infinite tree): {L.mean():.3f}")


# ---------------------------------------------------------------------------
# Synthetic traces: lets you develop/debug every stage before the GPU is ready.
# ---------------------------------------------------------------------------

def make_synthetic_traces(
    n_prompts: int = 200,
    steps_per_prompt: int = 100,
    n_heads: int = 5,
    topk: int = 10,
    dependence: float = 1.0,
    seed: int = 0,
) -> TraceSet:
    """Generate traces where head correctness is *positively dependent* across depth.

    A latent per-step "predictability" z is shared by all heads (e.g. boilerplate
    text is easy for every head at once). With dependence > 0, P(both heads right)
    exceeds the product of marginals, which is the failure mode the proposal's
    Step 1 diagnoses. dependence = 0 gives independent heads (sanity check: the
    conditional estimator should then match the marginal product).

    Head probabilities are informative but miscalibrated (overconfident for deep
    heads), so MEDUSA+ prob-product scoring is also imperfect.
    """
    rng = np.random.default_rng(seed)
    N, H, K = n_prompts * steps_per_prompt, n_heads, topk

    z = rng.normal(size=N)                                     # shared latent
    z_prompt = rng.normal(size=n_prompts).repeat(steps_per_prompt) * 0.5
    base = np.linspace(0.6, -1.6, H)                           # deeper heads are worse
    gold_rank = np.full((N, H), -1, dtype=np.int16)
    topk_prob = np.zeros((N, H, K), dtype=np.float32)

    for j in range(H):
        eps = rng.normal(size=N)
        skill = base[j] + dependence * (z + z_prompt) + (1.0 if dependence == 0 else 0.6) * eps
        p_top1 = 1 / (1 + np.exp(-skill))
        u = rng.random(N)
        # rank of gold: 0 w.p. p_top1, else geometric-ish over 1..K-1 or out of top-K
        # harder steps put the gold token further down (often outside top-K)
        r = np.where(u < p_top1, 0, rng.geometric(0.08 + 0.5 * p_top1))
        r = np.where(r >= K, -1, r)
        gold_rank[:, j] = r
        # observed head distribution: confidence tracks skill, with deep-head overconfidence
        conf_logit = skill + 0.4 * j * 0.5 + rng.normal(scale=0.7, size=N)
        top1 = 1 / (1 + np.exp(-conf_logit))
        tail = rng.dirichlet(np.ones(K - 1) * 0.6, size=N)
        tail = -np.sort(-tail, axis=1) * (1 - top1)[:, None] * 0.9
        topk_prob[:, j, 0] = top1
        topk_prob[:, j, 1:] = tail

    root_prob = (1 / (1 + np.exp(-(1.5 + dependence * z + rng.normal(scale=0.5, size=N))))).astype(np.float32)
    prompt_id = np.arange(n_prompts, dtype=np.int32).repeat(steps_per_prompt)
    position = np.tile(np.arange(steps_per_prompt, dtype=np.int32), n_prompts)
    return TraceSet(gold_rank, topk_prob, root_prob, prompt_id, position,
                    meta=dict(source="synthetic", dependence=dependence, seed=seed))
