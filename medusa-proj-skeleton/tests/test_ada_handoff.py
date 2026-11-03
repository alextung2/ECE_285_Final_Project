import importlib.util
import json
from pathlib import Path
import pickle
import sys
import types

import numpy as np
import pytest

from mcal.estimators import (Scorer, MedusaAccuracy, MedusaPlusProb, CountConditional,
                             LogisticConditional, DirectClassifier)
from mcal.features import conditional_nodes, candidate_pool_nodes, node_features, FEATURE_GROUPS
from mcal.handoff import split_calibration
from mcal.traces import make_synthetic_traces
from mcal.tree import select_tree, is_ancestor_closed


@pytest.fixture
def traces():
    return make_synthetic_traces(n_prompts=12, steps_per_prompt=12, seed=21)


def test_ada_feature_values_and_order_match_committed_source(traces):
    # Original Ada source remains a reference snapshot, not the active package.
    root = Path(__file__).resolve().parents[2] / 'mcal'
    package = types.ModuleType('ada_reference')
    package.__path__ = [str(root)]
    sys.modules['ada_reference'] = package
    try:
        spec = importlib.util.spec_from_file_location('ada_reference.features', root / 'features.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        nodes, _ = conditional_nodes(traces)
        for cap in (4, 9):
            for drop in [()] + [(g,) for g in FEATURE_GROUPS]:
                np.testing.assert_array_equal(node_features(traces, nodes, cap, drop),
                                              module.node_features(traces, nodes, cap, drop))
    finally:
        for name in list(sys.modules):
            if name.startswith('ada_reference'):
                del sys.modules[name]


def test_five_scorers_roundtrip_and_all_budgets(traces, tmp_path):
    ref = MedusaPlusProb()
    pool = candidate_pool_nodes(traces, ref, 16, n_steps=30)
    direct = DirectClassifier(rank_cap=9).fit(traces, pool=pool)
    union = candidate_pool_nodes(traces, [ref, direct], 16, n_steps=30)
    keys = [(int(s), tuple(p)) for s, p in zip(union[0].step, union[0].path)]
    assert len(set(keys)) == len(keys)
    scorers = [MedusaAccuracy().fit(traces), ref, CountConditional().fit(traces),
               LogisticConditional(rank_cap=9).fit(traces), DirectClassifier(rank_cap=9).fit(traces, pool=union)]
    for scorer in scorers:
        path = tmp_path / (scorer.name + '.pkl')
        scorer.save(path)
        loaded = Scorer.load(path)
        np.testing.assert_array_equal(scorer.path_scores(traces, pool[0]), loaded.path_scores(traces, pool[0]))
        for budget in (8, 16, 32, 63, 96):
            tree = select_tree(loaded, traces, 0 if loaded.dynamic else None, budget)
            assert len(tree) == budget and is_ancestor_closed(tree)


def test_legacy_feature_schema_cannot_silently_score(traces, tmp_path):
    scorer = LogisticConditional().fit(traces)
    del scorer.feature_schema
    path = tmp_path / 'legacy.pkl'
    path.write_bytes(pickle.dumps(scorer))
    with pytest.raises(ValueError, match='feature schema'):
        Scorer.load(path)
    with pytest.raises(ValueError, match='feature schema'):
        scorer.factor(traces, conditional_nodes(traces)[0])


@pytest.mark.parametrize('problem', ['overlap', 'missing', 'unknown', 'duplicate', 'empty'])
def test_manifest_fail_closed(traces, tmp_path, problem):
    train, val = list(range(8)), list(range(8, 12))
    if problem == 'overlap': val.append(0)
    if problem == 'missing': val.pop()
    if problem == 'unknown': val.append(99)
    if problem == 'duplicate': train.append(0)
    if problem == 'empty': train = []
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(dict(train_prompt_ids=train, validation_prompt_ids=val)))
    with pytest.raises(ValueError): split_calibration(traces, path)


def test_exact_manifest_and_real_requires_manifest(traces, tmp_path):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(dict(train_prompt_ids=list(range(8)), validation_prompt_ids=list(range(8, 12)))))
    traces.meta['source'] = 'real'
    fit, held = split_calibration(traces, path)
    assert set(fit.prompt_id) == set(range(8)) and set(held.prompt_id) == set(range(8, 12))
    with pytest.raises(ValueError, match='manifest'): split_calibration(traces)
