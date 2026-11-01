"""Decode boundary tests use controlled model outputs, not downloaded weights."""
import sys
import types
import importlib.util
from pathlib import Path
import pytest
torch = pytest.importorskip('torch')
from mcal.decode import StaticTree, medusa_greedy_generate, plain_greedy_generate

@pytest.fixture
def fake_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    def init_cache(model):
        return object(), None, None
    def update(ids, candidates, best, acc, retrieve, outputs, logits, medusa_logits,
               new_token, data, lengths):
        return torch.cat([ids, candidates[None, best, :acc+1]], dim=1), logits, medusa_logits, new_token+acc+1
    logits = torch.zeros(1,1,20)
    heads = torch.zeros(5,1,1,20)
    utilities = types.ModuleType('medusa.model.utils')
    utilities.reset_medusa_mode = lambda model: setattr(model.model, 'medusa_mask', None)
    utilities.initialize_medusa = lambda *a: (heads, logits)
    utilities.generate_medusa_buffers = lambda *a, **k: dict(medusa_attn_mask=None, tree_indices=None,
                                                            retrieve_indices=None, medusa_position_ids=None)
    utilities.generate_candidates = lambda *a, **k: (torch.tensor([[10,11,12,13]]), None)
    utilities.tree_decoding = lambda *a: (heads, logits, None)
    utilities.evaluate_posterior = lambda *a: (0, 3)
    utilities.update_inference_inputs = update
    cache = types.ModuleType('medusa.model.kv_cache')
    cache.initialize_past_key_values = init_cache
    monkeypatch.setitem(sys.modules, 'medusa.model.utils', utilities)
    monkeypatch.setitem(sys.modules, 'medusa.model.kv_cache', cache)
    class Model:
        device = torch.device('cpu')
        tokenizer = types.SimpleNamespace(eos_token_id=19)
        model = types.SimpleNamespace(medusa_mask=None)
        calls = 0
        def __init__(self):
            self.base_model = self
        def __call__(self, **kw):
            self.calls += 1
            out = torch.zeros(1,1,20)
            out[0,0,10] = 1
            return types.SimpleNamespace(logits=out)
    return Model()

def test_medusa_limit_and_inference_mode(fake_gpu):
    ids, st = medusa_greedy_generate(fake_gpu, torch.tensor([[1]]), StaticTree([[0]]), 2)
    assert ids.tolist() == [[1,10,11]]
    assert st.new_tokens == 2 and st.accept_lengths == [2] and st.raw_accept_lengths == [4]
    assert fake_gpu.model.medusa_mask is None

def test_medusa_stops_at_first_eos(fake_gpu):
    fake_gpu.tokenizer.eos_token_id = 11
    ids, st = medusa_greedy_generate(fake_gpu, torch.tensor([[1]]), StaticTree([[0]]), 8)
    assert ids.tolist() == [[1,10,11]] and st.steps == 1

def test_greedy_no_unused_final_forward(fake_gpu):
    ids, st = plain_greedy_generate(fake_gpu, torch.tensor([[1]]), 3)
    assert fake_gpu.calls == 3 and st.new_tokens == 3

def test_zero_output_limit(fake_gpu):
    for fn, args in [(plain_greedy_generate, ()), (medusa_greedy_generate, (StaticTree([[0]]),))]:
        ids, st = fn(fake_gpu, torch.tensor([[1]]), *args, max_new_tokens=0)
        assert ids.tolist() == [[1]] and st.steps == 0

def test_agreement_rejects_matching_prefix_with_different_length():
    path = Path(__file__).resolve().parents[1] / 'scripts'
    sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location('benchmark', path/'05_benchmark_speed.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert not mod.exact_agreement(torch.tensor([[1,2]]), torch.tensor([[1,2,3]]))
    assert mod.exact_agreement(torch.tensor([[1,2]]), torch.tensor([[1,2]]))
