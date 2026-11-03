import json
import numpy as np
import pytest
from mcal.precision import (precision_spec, require_compatible, trace_contract,
                            require_trace_config, promote_execution, ranked_probabilities)
from mcal.traces import TraceSet, make_synthetic_traces


def config(mode='nf4_fp16'):
    return {'model': dict(name='test-model', load_in_4bit=True, dtype='float16', precision_mode=mode)}


@pytest.mark.parametrize('mode,dtype', [('nf4_fp16', 'float16'), ('nf4_fp32_from_fp16', 'float32')])
def test_explicit_modes_keep_same_storage(mode, dtype):
    spec = precision_spec(config(mode)['model'])
    assert spec['compute_dtype'] == spec['head_dtype'] == dtype
    assert spec['weight_storage'] == 'nf4' and spec['initial_dtype'] == 'float16'


@pytest.mark.parametrize('key,value', [('dtype', 'float32'), ('load_in_4bit', False),
                                      ('precision_mode', '4bit_weights_fp32_execution')])
def test_ambiguous_or_conflicting_config_rejected(key, value):
    cfg = config()['model']
    cfg[key] = value
    with pytest.raises(ValueError):
        precision_spec(cfg)


def real_trace(mode='nf4_fp16'):
    ts = make_synthetic_traces(3, 5)
    ts.meta = dict(source='real', execution_contract=precision_spec(config(mode)['model']))
    return ts


def test_metadata_roundtrip_and_mixing(tmp_path):
    a = real_trace()
    path = tmp_path / 'trace.npz'
    a.save(path)
    b = TraceSet.load(path)
    require_compatible(trace_contract(a), trace_contract(b))
    assert TraceSet.concat([a, b]).n_steps == 30
    with pytest.raises(ValueError):
        TraceSet.concat([a, real_trace('nf4_fp32_from_fp16')])
    with pytest.raises(ValueError):
        TraceSet.concat([a, make_synthetic_traces(3, 5)])
    b.meta = {}
    with pytest.raises(ValueError):
        TraceSet.concat([a, b])
    with pytest.raises(ValueError):
        require_trace_config(a, config('nf4_fp32_from_fp16'))
    with pytest.raises(ValueError):
        require_trace_config(a, config(), synthetic=True)


def test_pilot_json_is_not_silently_converted(tmp_path):
    path = tmp_path / 'prompt_24.json'
    path.write_text(json.dumps({'nodes': [{'prefix_accepted': 0}]}))
    with pytest.raises(ValueError, match='missing full top-K'):
        TraceSet.load(path)


def test_promotion_preserves_packed_weights_and_quant_state():
    torch = pytest.importorskip('torch')
    class Packed(torch.nn.Parameter):
        pass
    class Linear4bit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = Packed(torch.tensor([1, 255], dtype=torch.uint8), requires_grad=False)
            self.quant_state = object()
            self.compute_dtype = torch.float16
            self.compute_type_is_set = False
    model = torch.nn.Module()
    model.backbone = Linear4bit()
    model.lm_head = torch.nn.Linear(2, 2).half()
    model.register_buffer('rotary_cache', torch.ones(2, dtype=torch.float16))
    model.register_buffer('indices', torch.tensor([1, 2]))
    packed, state = model.backbone.weight, model.backbone.quant_state
    promote_execution(model, torch.float32, Linear4bit, Packed)
    assert model.backbone.weight is packed and model.backbone.quant_state is state
    assert packed.tolist() == [1, 255] and packed.dtype == torch.uint8
    assert model.backbone.compute_dtype == torch.float32 and model.backbone.compute_type_is_set
    assert model.lm_head.weight.dtype == model.rotary_cache.dtype == torch.float32
    assert model.indices.dtype == torch.int64


def test_candidate_ranks_use_logits_even_when_softmax_underflows():
    torch = pytest.importorskip('torch')
    logits = torch.tensor([[1000., 3., 2., 1., 0.]], dtype=torch.float16)
    p, ids = ranked_probabilities(logits, 4)
    assert torch.equal(ids, logits.topk(4, -1).indices)
    assert ids.tolist() == [[0, 1, 2, 3]]
    assert torch.equal(p, torch.softmax(logits.float(), -1).gather(-1, ids))


def test_runtime_audit_rejects_fp16_head_in_fp32_mode():
    torch = pytest.importorskip('torch')
    from mcal.precision import audit_model
    class Packed(torch.nn.Parameter):
        pass
    class Linear4bit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = Packed(torch.tensor([1], dtype=torch.uint8), requires_grad=False)
            self.weight.quant_type = 'nf4'
            self.weight.compress_statistics = True
            self.weight.quant_state = object()
            self.compute_dtype = torch.float32
    model = torch.nn.Module()
    model.backbone = Linear4bit()
    model.lm_head = torch.nn.Linear(2, 2)
    model.medusa_head = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
    spec = precision_spec(config('nf4_fp32_from_fp16')['model'])
    with pytest.raises(RuntimeError, match='input hook'):
        audit_model(model, spec, Linear4bit, Packed)
    promote_execution(model, torch.float32, Linear4bit, Packed)
    assert audit_model(model, spec, Linear4bit, Packed)['linear4bit_count'] == 1
    model.lm_head.half()
    with pytest.raises(RuntimeError, match='Prediction-head dtype'):
        audit_model(model, spec, Linear4bit, Packed)


@pytest.mark.parametrize('source', ['precision_test', 'candidate_loader'])
def test_promotion_matches_lucia_committed_patches(monkeypatch, source):
    """Execute her actual promotion code on a CPU fixture, not a rewritten oracle."""
    import runpy
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    torch = pytest.importorskip('torch')
    repo = Path(__file__).resolve().parents[2]
    if source == 'precision_test':
        historical = runpy.run_path(str(repo / 'test_medusa_precision.py'))
        patch = historical['LM_HEAD_PATCH'] + historical['FP32_EXECUTION_PATCH']
    else:
        text = (repo / 'load_model_candidate_4bit_weights_fp32_execution.py').read_text()
        patch = text[text.index('# Compute the backbone output projection in FP32.'):]
        patch = patch.split('\nprecision_mode =', 1)[0]

    class Packed(torch.nn.Parameter):
        pass
    class Linear4bit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = Packed(torch.tensor([1, 255], dtype=torch.uint8), requires_grad=False)
            self.quant_state = {'scale': torch.tensor([0.5])}
            self.compute_dtype = torch.float16
            self.compute_type_is_set = False

    # Share the floating initialization; make independent packed-weight objects.
    def model():
        m = torch.nn.Module()
        m.backbone = Linear4bit()
        m.lm_head = torch.nn.Linear(2, 3).half()
        m.medusa_head = torch.nn.ModuleList([torch.nn.Linear(2, 3).half()])
        m.register_buffer('rotary_cache', torch.tensor([0.3], dtype=torch.float16))
        m.register_buffer('indices', torch.tensor([1, 2]))
        return m
    reference, actual = model(), model()
    actual.load_state_dict(reference.state_dict())
    before = [(m.backbone.weight.data_ptr(), m.backbone.quant_state) for m in (reference, actual)]
    bnb = SimpleNamespace(nn=SimpleNamespace(Linear4bit=Linear4bit, Params4bit=Packed))
    monkeypatch.setitem(sys.modules, 'bitsandbytes', bnb)
    exec(patch, {'model': reference, 'torch': torch, 'bnb': bnb})
    promote_execution(actual, torch.float32, Linear4bit, Packed)
    for m, (pointer, state) in zip((reference, actual), before):
        assert m.backbone.weight.data_ptr() == pointer
        assert m.backbone.quant_state is state
        assert m.backbone.compute_dtype == torch.float32
        assert m.backbone.compute_type_is_set
    assert reference.state_dict().keys() == actual.state_dict().keys()
    for name, tensor in reference.state_dict().items():
        assert tensor.dtype == actual.state_dict()[name].dtype
        assert torch.equal(tensor, actual.state_dict()[name])
    half_input = torch.tensor([[0.25, 0.5]], dtype=torch.float16)
    assert torch.equal(reference.lm_head(half_input), actual.lm_head(half_input))
    assert actual.lm_head(half_input).dtype == torch.float32
    promote_execution(actual, torch.float32, Linear4bit, Packed)
    assert len(actual.lm_head._forward_pre_hooks) == 1  # idempotent, like Lucia's patch


def test_pre_alignment_contract_is_not_silently_accepted():
    ts = real_trace('nf4_fp32_from_fp16')
    del ts.meta['execution_contract']['lm_head_fp32_input_hook']
    with pytest.raises(ValueError, match='mismatch'):
        require_trace_config(ts, config('nf4_fp32_from_fp16'))


def test_observer_checks_node_labels_not_only_final_tokens():
    torch = pytest.importorskip('torch')
    from mcal.validation import TraceObserver
    logits = torch.tensor([[2., 1., 0.], [2., 1., 0.]])
    probs, ids = ranked_probabilities(logits, 2)
    ts = TraceSet(np.array([[0, 1]]), probs.numpy()[None], np.array([1.]),
                  np.array([0]), np.array([0]), topk_token=ids.numpy()[None])
    observer = TraceObserver(ts, torch.tensor([[9, 2, 0, 1]]), 1)
    buffers = {'retrieve_indices': torch.tensor([[0, 1, 2]])}
    candidates = torch.tensor([[2, 0, 1]])
    verified = torch.tensor([[[4., 0., 0.], [0., 4., 0.], [0., 0., 4.]]])
    observer(torch.tensor([[9]]), [[0], [0, 1]], buffers, candidates, verified, logits)
    assert observer.passed
    verified[0, 1] = torch.tensor([4., 0., 0.])
    observer(torch.tensor([[9]]), [[0], [0, 1]], buffers, candidates, verified, logits)
    assert not observer.passed
    assert observer.rows[-1]['label_mismatches'] == [[0, 1]]


@pytest.mark.parametrize('failure', ['generation', 'teacher_tail'])
def test_collection_rejects_disagreement_even_outside_labelled_roots(monkeypatch, failure):
    torch = pytest.importorskip('torch')
    import sys
    from types import SimpleNamespace
    import mcal.collect as collect
    monkeypatch.setitem(sys.modules, 'medusa.model.utils',
                        SimpleNamespace(reset_medusa_mode=lambda model: None))
    monkeypatch.setattr(collect, 'default_medusa_choices', lambda: [[0]])
    full = torch.tensor([[9, 0, 1, 2, 3]])
    reference = full.clone()
    if failure == 'generation':
        reference[0, -1] = 2
    monkeypatch.setattr(collect, 'medusa_greedy_generate', lambda *a, **k: (full, None))
    monkeypatch.setattr(collect, 'plain_greedy_generate', lambda *a, **k: (reference, None))
    class Model:
        medusa = 1
        base_model = SimpleNamespace(device='cpu')
        def __call__(self, *args, **kwargs):
            logits = torch.zeros(1, 5, 10)
            for i in range(4):
                logits[0, i, i] = 10
            if failure == 'teacher_tail':
                # Only the tail is wrong: t_idx stops at 2, tail prediction is 3.
                logits[0, 3, 4] = 20
            return torch.zeros(1, 1, 5, 10), None, logits
    tok = lambda *a, **k: SimpleNamespace(input_ids=torch.tensor([[9]]))
    with pytest.raises(RuntimeError, match='mismatch'):
        collect.traces_for_prompt(Model(), tok, 'test', 24, topk=2)
