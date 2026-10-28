"""Online greedy Medusa decoding with a pluggable tree policy (proposal Weeks 6-7).

This is a trimmed copy of Medusa's `medusa_generate` loop (temperature = 0,
batch size 1) with one change: the tree (`medusa_choices`) can be rebuilt at
every step by a `TreePolicy`. That is exactly what we need to measure
  * accepted tokens per verification step (online, not trace-simulated),
  * tree-building overhead,
  * end-to-end tokens/sec vs. ordinary greedy decoding (the 1.0x reference),
  * agreement with greedy output (should be exact for greedy verification).

Uses medusa.model.utils (checked against FasterDecoding/Medusa main):
  generate_medusa_buffers, initialize_medusa, reset_medusa_mode,
  generate_candidates, tree_decoding, evaluate_posterior, update_inference_inputs
and medusa.model.kv_cache.initialize_past_key_values.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .traces import TraceSet
from .tree import select_tree, tree_to_medusa_choices


# ---------------------------------------------------------------------------
# Tree policies
# ---------------------------------------------------------------------------

class StaticTree:
    """Fixed medusa_choices (original Medusa, or our count-based fixed tree)."""

    def __init__(self, medusa_choices):
        self.choices = [list(c) for c in medusa_choices]
        self.dynamic = False

    def __call__(self, medusa_logits_last, logits_last):
        return self.choices


class DynamicTree:
    """Per-step tree from any mcal Scorer (MEDUSA+, direct classifier, ours).

    Converts the live head distributions into a 1-step TraceSet so the exact
    same select_tree() code path is used online and on traces.
    """

    def __init__(self, scorer, budget: int, topk: int = 10):
        self.scorer, self.budget, self.topk = scorer, budget, topk
        self.dynamic = True

    def __call__(self, medusa_logits_last, logits_last):
        import torch
        hp = torch.softmax(medusa_logits_last.float(), -1)            # [H, V]
        top_p = hp.topk(self.topk, -1).values.cpu().numpy()[None]     # [1, H, K]
        root = torch.softmax(logits_last.float(), -1).max().item()
        H = top_p.shape[1]
        ts = TraceSet(gold_rank=np.zeros((1, H), np.int16), topk_prob=top_p.astype(np.float32),
                      root_prob=np.array([root], np.float32), prompt_id=np.zeros(1, np.int32),
                      position=np.zeros(1, np.int32))
        return tree_to_medusa_choices(select_tree(self.scorer, ts, 0, self.budget))


# ---------------------------------------------------------------------------
# Decoding loops
# ---------------------------------------------------------------------------

@dataclass
class DecodeStats:
    new_tokens: int = 0
    steps: int = 0
    wall_s: float = 0.0
    tree_s: float = 0.0            # policy + buffer construction
    accept_lengths: list = field(default_factory=list)   # accepted tokens per step (incl. g_0)

    @property
    def tokens_per_s(self):
        return self.new_tokens / max(self.wall_s, 1e-9)

    @property
    def mean_accepted(self):
        return float(np.mean(self.accept_lengths)) if self.accept_lengths else float("nan")


def medusa_greedy_generate(model, input_ids, policy, max_new_tokens: int = 256):
    """Returns (output_ids [1, L], DecodeStats). Greedy verification only."""
    import torch
    from medusa.model.kv_cache import initialize_past_key_values
    from medusa.model.utils import (evaluate_posterior, generate_candidates, generate_medusa_buffers,
                                    initialize_medusa, reset_medusa_mode, tree_decoding,
                                    update_inference_inputs)

    assert input_ids.shape[0] == 1
    dev = model.base_model.device
    input_ids = input_ids.clone().to(dev)
    stats = DecodeStats()
    cache = {}

    def buffers_for(choices):
        key = tuple(tuple(c) for c in choices)
        if key not in cache:
            if len(cache) > 256:
                cache.clear()
            cache[key] = generate_medusa_buffers(choices, device=dev)
        return cache[key]

    pkv, pkv_data, cur_len = initialize_past_key_values(model.base_model)
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    reset_medusa_mode(model)
    # prefill with a plain causal mask; the tree mask is installed per step below
    medusa_logits, logits = initialize_medusa(input_ids, model, None, pkv)
    input_len, new_token = input_ids.shape[1], 0

    while new_token < max_new_tokens:
        t0 = time.perf_counter()
        choices = policy(medusa_logits[:, 0, -1], logits[0, -1])
        bufs = buffers_for(choices)
        model.base_model.model.medusa_mask = bufs["medusa_attn_mask"]
        stats.tree_s += time.perf_counter() - t0

        candidates, tree_candidates = generate_candidates(
            medusa_logits, logits, bufs["tree_indices"], bufs["retrieve_indices"], temperature=0.0)
        medusa_logits, logits, outputs = tree_decoding(
            model, tree_candidates, pkv, bufs["medusa_position_ids"], input_ids, bufs["retrieve_indices"])
        best, acc_len = evaluate_posterior(logits, candidates, 0.0)
        input_ids, logits, medusa_logits, new_token = update_inference_inputs(
            input_ids, candidates, best, acc_len, bufs["retrieve_indices"], outputs,
            logits, medusa_logits, new_token, pkv_data, cur_len)
        stats.steps += 1
        stats.accept_lengths.append(int(acc_len) + 1)
        if model.tokenizer.eos_token_id in input_ids[0, input_len:]:
            break

    torch.cuda.synchronize()
    stats.wall_s = time.perf_counter() - t_start
    stats.new_tokens = input_ids.shape[1] - input_len
    return input_ids, stats


def plain_greedy_generate(model, input_ids, max_new_tokens: int = 256):
    """1.0x speed reference: one backbone forward per token, same KV cache impl."""
    import torch
    from medusa.model.kv_cache import initialize_past_key_values
    from medusa.model.utils import reset_medusa_mode

    dev = model.base_model.device
    ids = input_ids.clone().to(dev)
    reset_medusa_mode(model)
    pkv, _, _ = initialize_past_key_values(model.base_model)
    stats = DecodeStats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.base_model(input_ids=ids, past_key_values=pkv)
        for _ in range(max_new_tokens):
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            stats.steps += 1
            stats.accept_lengths.append(1)
            if nxt.item() == model.tokenizer.eos_token_id:
                break
            out = model.base_model(input_ids=nxt, past_key_values=pkv)
    torch.cuda.synchronize()
    stats.wall_s = time.perf_counter() - t0
    stats.new_tokens = ids.shape[1] - input_ids.shape[1]
    return ids, stats
