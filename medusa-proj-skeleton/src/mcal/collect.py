"""Collect verification traces from the real model (proposal Week 2).

Recipe per prompt:
  1. Greedy-generate a response with our Medusa loop (decode.py) at temperature
     0 using the default Medusa tree. Greedy verification is lossless, so this is
     the backbone's greedy output, just faster.
  2. One teacher-forced forward over prompt + response with medusa_forward=True.
     At every position t we get the backbone logits and all H head logits from
     the same hidden state Medusa would use online.
  3. lm_head at t drafts t + 1 (= g_0, the root); head j (0-based) drafts
     t + j + 2, so gold_rank[t, j] = rank of token[t + j + 2] in head j's top-K.

Built-in sanity check: argmax of the teacher-forced backbone logits should
reproduce the generated tokens (`greedy_agree` ~ 1.0). If it doesn't, the
cached (generation) and uncached (teacher-forced) numerics differ, e.g. under
4-bit; investigate before trusting labels.
"""

from __future__ import annotations

import numpy as np

from .data import vicuna_prompt
from .decode import StaticTree, medusa_greedy_generate
from .traces import TraceSet


def default_medusa_choices():
    from medusa.model.medusa_choices import vicuna_7b_stage2
    return vicuna_7b_stage2


def traces_for_prompt(model, tokenizer, prompt: str, prompt_id: int, topk: int = 10,
                      max_new_tokens: int = 256, keep_tokens: bool = False) -> tuple[TraceSet, float]:
    import torch
    from medusa.model.utils import reset_medusa_mode

    dev = model.base_model.device
    prompt_ids = tokenizer([vicuna_prompt(prompt)], return_tensors="pt").input_ids.to(dev)
    full, _ = medusa_greedy_generate(model, prompt_ids, StaticTree(default_medusa_choices()),
                                     max_new_tokens=max_new_tokens)
    P, T, H = prompt_ids.shape[1], full.shape[1], model.medusa

    reset_medusa_mode(model)                    # ordinary causal mask, no KV cache
    with torch.inference_mode():
        # TODO(week 1): confirm KVLlama accepts past_key_values=None here
        # (it should: past_key_values_length defaults to 0).
        medusa_logits, _, logits = model(full, output_orig=True, medusa_forward=True)

    t_idx = torch.arange(P - 1, T - H - 1, device=dev)   # positions with all H drafts labelled
    if len(t_idx) == 0:
        raise ValueError("response too short to label")
    tgt = full[0]

    root_prob = torch.softmax(logits[0, t_idx].float(), -1).gather(-1, tgt[t_idx + 1, None]).squeeze(-1)
    greedy_agree = (logits[0, t_idx].argmax(-1) == tgt[t_idx + 1]).float().mean().item()

    top_p, top_tok = torch.softmax(medusa_logits[:, 0, t_idx].float(), -1).topk(topk, dim=-1)  # [H, n, K]
    gold = torch.stack([tgt[t_idx + j + 2] for j in range(H)], 0)                            # [H, n]
    match = top_tok == gold[..., None]
    rank = torch.where(match.any(-1), match.float().argmax(-1).long(), torch.full_like(gold, -1))

    ts = TraceSet(
        gold_rank=rank.T.cpu().numpy().astype(np.int16),
        topk_prob=top_p.permute(1, 0, 2).cpu().numpy().astype(np.float32),
        root_prob=root_prob.cpu().numpy().astype(np.float32),
        prompt_id=np.full(len(t_idx), prompt_id, dtype=np.int32),
        position=(t_idx - (P - 1)).cpu().numpy().astype(np.int32),
        topk_token=top_tok.permute(1, 0, 2).cpu().numpy().astype(np.int32) if keep_tokens else None,
    )
    return ts, greedy_agree
