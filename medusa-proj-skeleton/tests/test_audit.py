import numpy as np
import pytest
from mcal.audit import validate_arrays, audit_file, audit_pair
from mcal.precision import precision_spec
from mcal.traces import TraceSet


def fixture():
    cfg = dict(model=dict(name='test', load_in_4bit=True, precision_mode='nf4_fp32_from_fp16'),
               traces=dict(topk=2))
    contract = dict(precision_spec(cfg['model']), packages={'torch': 'test'},
                    medusa_source_sha256={'utils.py': 'test'}, loaded_heads_sha256='test', base_revision='test')
    ts = TraceSet(np.array([[0, -1], [1, 0]]),
                  np.array([[[.6, .2], [.5, .1]], [[.7, .1], [.4, .3]]]),
                  np.array([.5, .6]), np.array([1, 2]), np.array([0, 0]),
                  meta=dict(source='real', execution_contract=contract, dataset='alpacaeval', dataset_revision='test'),
                  topk_token=np.array([[[4, 5], [6, 7]], [[4, 5], [6, 7]]]))
    return ts, cfg


def test_valid_npz_is_read_only_and_auditable(tmp_path):
    ts, cfg = fixture()
    path = tmp_path / 'traces.npz'
    ts.save(path)
    before = path.read_bytes()
    loaded, report = audit_file(path, cfg)
    assert report['steps'] == 2 and not report['warnings']
    assert loaded.gold_rank.tolist() == [[0, -1], [1, 0]]
    assert path.read_bytes() == before


@pytest.mark.parametrize('failure', ['nan', 'mass', 'rank', 'duplicate_state', 'duplicate_token', 'shape', 'fractional_label'])
def test_corrupt_arrays_rejected(failure):
    ts, _ = fixture()
    if failure == 'nan': ts.root_prob[0] = np.nan
    if failure == 'mass': ts.topk_prob[0, 0] = [.8, .7]
    if failure == 'rank': ts.gold_rank[0, 0] = 2
    if failure == 'duplicate_state': ts.prompt_id[1] = 1
    if failure == 'duplicate_token': ts.topk_token[0, 0] = [4, 4]
    if failure == 'shape': ts.position = np.array([0])
    if failure == 'fractional_label': ts.gold_rank = ts.gold_rank.astype(float) + .5
    with pytest.raises(ValueError): validate_arrays(ts)


def test_same_dataset_leakage_rejected_but_cross_dataset_ids_not_conflated():
    calib, _ = fixture()
    ev, _ = fixture()
    with pytest.raises(ValueError, match='overlap'): audit_pair(calib, ev)
    ev.meta['dataset'] = 'mtbench'
    assert 'not comparable' in audit_pair(calib, ev)
    ev.meta['execution_contract']['compute_dtype'] = 'float16'
    with pytest.raises(ValueError, match='Incompatible'): audit_pair(calib, ev)


def test_missing_runtime_provenance_rejected(tmp_path):
    ts, cfg = fixture()
    del ts.meta['execution_contract']['loaded_heads_sha256']
    path = tmp_path / 'trace.npz'
    ts.save(path)
    with pytest.raises(ValueError, match='runtime provenance'): audit_file(path, cfg)
