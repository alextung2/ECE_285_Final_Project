"""Regression tests for Alex's tree-selection and confidence-interval contract."""
import itertools
import json
import numpy as np
import pytest
from mcal.estimators import MedusaPlusProb
from mcal.evaluate import paired_bootstrap
from mcal.traces import make_synthetic_traces
from mcal.tree import select_tree, validate_choices, tree_to_medusa_choices

@pytest.mark.parametrize('budget', [8,16,32,63,96])
def test_all_required_budgets_and_json_roundtrip(budget):
    ts = make_synthetic_traces(2, 4, seed=17)
    tree = select_tree(MedusaPlusProb(), ts, 0, budget)
    choices = json.loads(json.dumps(tree_to_medusa_choices(tree)))
    assert len(validate_choices(choices)) == budget
    score = dict(tree)
    assert all(v <= score.get(p[:-1], 1.0) for p,v in tree)

def test_best_first_matches_exhaustive_top_scores():
    ts = make_synthetic_traces(2, 4, n_heads=3, topk=3, seed=2)
    exhaustive = []
    for d in range(1,4):
        for path in itertools.product(range(3), repeat=d):
            exhaustive.append(float(np.prod([float(ts.topk_prob[0,j,r]) for j,r in enumerate(path)])))
    tree = select_tree(MedusaPlusProb(), ts, 0, 13)
    np.testing.assert_allclose([s for _,s in tree], sorted(exhaustive, reverse=True)[:13])

@pytest.mark.parametrize('choices', [[[0,1]], [[0],[0]], [[10]], [[]], [[-1]]])
def test_invalid_choices_rejected(choices):
    with pytest.raises(ValueError):
        validate_choices(choices)

def test_bad_scores_and_impossible_budget_rejected():
    ts = make_synthetic_traces(2, 4, n_heads=1, seed=3)
    with pytest.raises(ValueError):
        select_tree(MedusaPlusProb(), ts, 0, 11)
    class Bad:
        def child_scores(self, *args):
            return np.full(10, 1.1)
    with pytest.raises(ValueError, match='parent score'):
        select_tree(Bad(), ts, 0, 1)

def test_prompt_bootstrap_preserves_cluster_uncertainty():
    a = np.repeat([1.0, 5.0], 100)
    mean, lo, hi = paired_bootstrap(a, np.zeros_like(a), np.repeat([0,1],100))
    assert mean == 3 and lo == 1 and hi == 5

def test_choices_load_in_real_medusa_buffers():
    pytest.importorskip('medusa')
    from medusa.model.utils import generate_medusa_buffers
    ts = make_synthetic_traces(2, 4, seed=17)
    for budget in [8,16,32,63,96]:
        choices = tree_to_medusa_choices(select_tree(MedusaPlusProb(), ts, 0, budget))
        buffers = generate_medusa_buffers(choices, device='cpu')
        assert buffers['medusa_attn_mask'].shape[-1] == budget + 1
        assert buffers['tree_indices'].numel() == budget + 1

def test_prompt_matches_notebook_vicuna_template():
    pytest.importorskip('fastchat')
    from fastchat.conversation import get_conv_template
    from mcal.data import vicuna_prompt
    question = 'Explain what a robot is in one sentence.'
    conv = get_conv_template('vicuna_v1.1')
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    assert vicuna_prompt(question) == conv.get_prompt()
