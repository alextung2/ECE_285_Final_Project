import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mcal.provenance import (read_prompt_records, prompt_digest, require_disjoint_text,
                             check_prompt_binding, check_collection_status, exclude_evaluation_overlap)
from mcal.traces import make_synthetic_traces, TraceSet


def records():
    return [dict(prompt_id=i, instruction=f"Question {i}") for i in range(3)]


def test_prompt_hash_and_coverage_bind_exact_inputs(tmp_path):
    ts = make_synthetic_traces(3, 2)
    recs = records()
    ts.meta["prompt_sha256"] = prompt_digest(recs)
    assert check_prompt_binding(ts, recs)["exact_coverage"]
    with pytest.raises(ValueError, match="hash"):
        check_prompt_binding(ts, list(reversed(recs)))
    with pytest.raises(ValueError, match="coverage"):
        check_prompt_binding(ts, recs[:2])
    recs[0]["instruction"] = "Different prompt"
    with pytest.raises(ValueError, match="hash"):
        check_prompt_binding(ts, recs)


@pytest.mark.parametrize("bad", [[], [{"prompt_id": True, "instruction": "x"}],
    [{"prompt_id": 0, "instruction": ""}], [{"prompt_id": 1, "instruction": "a"}] * 2])
def test_invalid_prompt_inputs_rejected(tmp_path, bad):
    p = tmp_path / "prompts.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        read_prompt_records(p)


def test_cross_dataset_text_overlap_despite_different_ids():
    with pytest.raises(ValueError, match="text overlap"):
        require_disjoint_text([dict(prompt_id=1, instruction="Hello  WORLD")],
                              [dict(prompt_id=900, instruction=" hello\nworld ")])
    assert "semantic overlap not assessed" in require_disjoint_text(records()[:1], records()[1:])


def test_precollection_exclusion_preserves_originals_and_audit_trail():
    train = records()
    evaluation = [dict(prompt_id=81, instruction="QUESTION 1")]
    before = json.dumps(train)
    kept, excluded = exclude_evaluation_overlap(train, evaluation)
    assert [r["prompt_id"] for r in kept] == [0, 2]
    assert excluded[0]["prompt_id"] == 1
    assert excluded[0]["evaluation_prompt_ids"] == [81]
    assert json.dumps(train) == before
    require_disjoint_text(kept, evaluation)


@pytest.mark.parametrize("metadata", [
    {"collection_complete": False}, {"missing_prompt_ids": [99]},
    {"expected_prompt_ids": [0, 1, 2, 3]}, {"expected_prompt_ids": [0, 1, 1, 2]},
])
def test_incomplete_collection_cannot_be_certified(metadata):
    ts = make_synthetic_traces(3, 2)
    ts.meta.update(metadata)
    with pytest.raises(ValueError):
        check_collection_status(ts)


def collector_module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("collector_cli_test", scripts / "01_collect_traces.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("fails", [False, True])
def test_collector_records_completion_and_never_silently_skips(monkeypatch, tmp_path, fails):
    import mcal.collect as collect
    from mcal.config import load_config
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    cfg["out_dir"] = str(tmp_path)
    recs = [dict(prompt_id=i, instruction=f"q{i}") for i in range(21)]
    prompt_file = tmp_path / "input.json"
    prompt_file.write_text(json.dumps(recs))
    cfg["traces"]["calib"]["prompt_file"] = str(prompt_file)
    mod = collector_module(monkeypatch)
    monkeypatch.setattr(mod, "parse", lambda *a: (SimpleNamespace(synthetic=False, split="calib"), cfg))
    monkeypatch.setitem(sys.modules, "mcal.model",
                        SimpleNamespace(load_medusa=lambda cfg: (SimpleNamespace(execution_contract={}), None)))
    def fake_collect(model, tok, prompt, prompt_id, *args):
        if fails and prompt_id == 20:
            raise ValueError("response too short to label")
        ts = make_synthetic_traces(1, 2)
        ts.prompt_id[:] = prompt_id
        return ts, 1.
    monkeypatch.setattr(collect, "traces_for_prompt", fake_collect)
    path = tmp_path / "traces/calib.npz"
    if fails:
        with pytest.raises(ValueError, match="too short"):
            mod.main()
        ts = TraceSet.load(path)
        assert ts.meta["collection_complete"] is False
        failure = json.loads(path.with_suffix(".failure.json").read_text())
        assert failure["prompt_id"] == 20 and failure["completed_prompt_ids"] == list(range(20))
        with pytest.raises(ValueError, match="Incomplete"):
            check_collection_status(ts)
    else:
        mod.main()
        ts = TraceSet.load(path)
        assert ts.meta["collection_complete"] is True
        check_collection_status(ts)
        assert check_prompt_binding(ts, recs)["exact_coverage"]
    assert read_prompt_records(path.with_suffix(".prompts.json")) == recs
