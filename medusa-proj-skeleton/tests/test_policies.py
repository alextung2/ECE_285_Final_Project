import json
import pytest
from mcal.estimators import LogisticConditional
from mcal.policies import load_policy
from mcal.traces import make_synthetic_traces
from mcal.precision import trace_contract


def test_static_and_learned_policy_handoffs_share_contract_checks(tmp_path):
    ts = make_synthetic_traces(4, 6)
    contract = trace_contract(ts)
    tree = tmp_path / "fixed_tree_count_cond_B2.json"
    tree.write_text(json.dumps(dict(choices=[[0], [1]], execution_contract=contract)))
    policy = load_policy("count_cond", 2, contract, tmp_path)
    assert policy.choices == [[0], [1]] and not policy.dynamic
    scorer = LogisticConditional().fit(ts)
    scorer.execution_contract = contract
    scorer.save(tmp_path / "logistic_cond.pkl")
    assert load_policy("logistic_cond", 2, contract, tmp_path).dynamic
    for name in ("count_cond", "logistic_cond"):
        with pytest.raises(ValueError, match="Incompatible"):
            load_policy(name, 2, {"precision_mode": "different"}, tmp_path)
    tree.write_text(json.dumps(dict(choices=[[0]], execution_contract=contract)))
    with pytest.raises(ValueError, match="draft nodes"):
        load_policy("count_cond", 2, contract, tmp_path)
    tree.write_text("[[0], [1]]")
    with pytest.raises(ValueError, match="Legacy"):
        load_policy("count_cond", 2, contract, tmp_path)
