"""Fast tests for the numpy core. Run: pytest -q"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcal.estimators import (CountConditional, DirectClassifier, LogisticConditional,  # noqa: E402
                             MedusaAccuracy, MedusaPlusProb)
from mcal.evaluate import accepted_tokens, evaluate_tree_policy  # noqa: E402
from mcal.features import NodeBatch, candidate_pool_nodes, conditional_nodes, node_features  # noqa: E402
from mcal.traces import TraceSet, make_synthetic_traces  # noqa: E402
from mcal.tree import fixed_tree, is_ancestor_closed, select_tree, tree_to_medusa_choices  # noqa: E402


@pytest.fixture(scope="module")
def ts():
    return make_synthetic_traces(n_prompts=40, steps_per_prompt=50, seed=3)


def test_roundtrip(tmp_path, ts):
    ts.save(tmp_path / "t.npz")
    t2 = TraceSet.load(tmp_path / "t.npz")
    assert np.array_equal(t2.gold_rank, ts.gold_rank) and t2.meta == ts.meta


def test_split_by_prompt_disjoint(ts):
    a, b = ts.split_by_prompt(0.7)
    assert not set(a.prompt_id) & set(b.prompt_id)
    assert a.n_steps + b.n_steps == ts.n_steps


def test_is_accepted_matches_definition(ts):
    s = 5
    gold = ts.gold_rank[s]
    path = np.full((1, ts.n_heads), -1, np.int16)
    L = ts.accept_chain_len()[s]
    if L >= 2:
        path[0, :2] = gold[:2]
        assert ts.is_accepted(np.array([s]), path, np.array([2]))[0]
        path[0, 1] = (gold[1] + 1) % ts.topk
        assert not ts.is_accepted(np.array([s]), path, np.array([2]))[0]


def test_conditional_nodes_only_accepted_parents(ts):
    nb, y = conditional_nodes(ts)
    deep = nb.depth > 1
    parent_path = nb.path.copy()
    parent_path[np.arange(len(nb)), nb.depth - 1] = -1
    assert ts.is_accepted(nb.step[deep], parent_path[deep], nb.depth[deep] - 1).all()
    # exactly one accepted child per accepted parent whose gold child is in top-K
    assert y.sum() <= len(np.unique(np.stack([nb.step, nb.depth], 1), axis=0))


@pytest.mark.parametrize("cls", [MedusaPlusProb, LogisticConditional, CountConditional, MedusaAccuracy])
def test_tree_is_ancestor_closed_and_budgeted(ts, cls):
    sc = cls().fit(ts)
    for B in [1, 7, 30]:
        tree = select_tree(sc, ts, 0, B) if sc.dynamic else fixed_tree(sc, ts, B)
        assert len(tree) == B and is_ancestor_closed(tree)
        scores = [s for _, s in tree]
        assert all(0 <= s <= 1 for s in scores)
        assert scores == sorted(scores, reverse=True)   # best-first => monotone pops


def test_medusa_choices_format(ts):
    ch = tree_to_medusa_choices(fixed_tree(MedusaAccuracy().fit(ts), ts, 10))
    assert ch[0] == [0] and all(isinstance(c, list) for c in ch)


def test_accepted_tokens_counts_chain(ts):
    s = int(np.argmax(ts.accept_chain_len()))
    gold = ts.gold_rank[s]
    full = {tuple(int(g) for g in gold[:d]) for d in range(1, ts.n_heads + 1)}
    assert accepted_tokens(ts, s, full) == 1 + ts.accept_chain_len()[s]
    assert accepted_tokens(ts, s, set()) == 1


def test_count_cond_matches_empirical_when_unsmoothed(ts):
    sc = CountConditional(alpha=1e-9).fit(ts)
    nb, y = conditional_nodes(ts)
    m = (nb.depth == 2) & (nb.parent_rank == 0) & (nb.own_rank == 0)
    assert np.isclose(sc.factor(ts, NodeBatch(nb.step[m], nb.path[m], nb.depth[m]))[0], y[m].mean())


def test_independent_heads_conditional_close_to_product():
    t = make_synthetic_traces(200, 100, dependence=0.0, seed=1)
    acc = MedusaAccuracy().fit(t)
    cond = CountConditional(alpha=1e-9).fit(t)
    # depth-2 top-1/top-1: P(S | parent) ~ marginal accuracy of head 2 when independent
    assert abs(cond.q[1, 0, 0] - acc.acc[1, 0]) < 0.03


def test_dependent_heads_product_underestimates():
    t = make_synthetic_traces(200, 100, dependence=1.0, seed=1)
    acc = MedusaAccuracy().fit(t)
    cond = CountConditional(alpha=1e-9).fit(t)
    assert cond.q[1, 0, 0] > acc.acc[1, 0] + 0.03


def test_features_shape_and_finite(ts):
    nb, _ = conditional_nodes(ts)
    X = node_features(ts, nb)
    assert X.shape[0] == len(nb) and np.isfinite(X).all()


def test_direct_and_path_scores_monotone(ts):
    pool = candidate_pool_nodes(ts, MedusaPlusProb(), 16, n_steps=200)
    d = DirectClassifier().fit(ts, pool=pool)
    tree = select_tree(d, ts, 3, 20)
    assert is_ancestor_closed(tree)
    p = d.path_scores(ts, pool[0])
    assert ((0 <= p) & (p <= 1)).all()


def test_evaluate_runs(ts):
    r = evaluate_tree_policy(MedusaPlusProb(), ts, 16, steps=np.arange(50))
    assert 1.0 <= r["mean_accepted"] <= 1 + ts.n_heads
