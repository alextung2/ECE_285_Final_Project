"""Validate exact tokens and acceptance lengths against official Medusa.
Uses the five prompts and 64-token limit from notebook cell 11.
"""
import json
import ast
import re
from pathlib import Path
from unittest.mock import patch
from _common import parse

PROMPTS = [
    'Explain what a robot is in one sentence.',
    'What is the difference between weather and climate?',
    'Explain why the sky appears blue.',
    "Translate 'Good morning, how are you?' into French.",
    'Write a Python function that returns the square of a number.',
]


def main():
    args, cfg = parse(__doc__)
    import torch
    import medusa.model.medusa_model as upstream
    from mcal.collect import default_medusa_choices
    from mcal.config import out_path
    from mcal.data import vicuna_prompt
    from mcal.decode import StaticTree, medusa_greedy_generate, plain_greedy_generate
    from mcal.model import load_medusa
    model, tok = load_medusa(cfg)
    from huggingface_hub import hf_hub_download
    saved = torch.load(hf_hub_download(cfg['model']['name'], 'medusa_lm_head.pt'),
                       map_location='cpu', weights_only=True)
    actual_heads = model.medusa_head.state_dict()
    if saved.keys() != actual_heads.keys():
        raise RuntimeError('Trained Medusa head keys do not match')
    for name, parameter in actual_heads.items():
        if not torch.equal(parameter, saved[name].to(device=parameter.device, dtype=parameter.dtype)):
            raise RuntimeError(f'Trained Medusa head mismatch: {name}')
    del saved, actual_heads
    print('Trained Medusa head weights verified.', flush=True)
    rows, limit = [], 64
    notebook = Path(__file__).resolve().parents[2] / 'ece285_Medusa_Project.ipynb'
    saved_lengths = []
    if notebook.exists():
        cells = json.loads(notebook.read_text(encoding='utf-8'))['cells']
        text = ''.join(''.join(o.get('text', [])) for o in cells[11].get('outputs', []))
        saved_lengths = [ast.literal_eval(x) for x in re.findall(
            r'Tokens accepted per round: (\[[^\]]*\])', text)]
    choices = default_medusa_choices()
    for prompt in PROMPTS:
        ids = tok([vicuna_prompt(prompt)], return_tensors='pt').input_ids.to(model.device)
        tokens, lengths, raw_lengths = [], [], []
        original = upstream.update_inference_inputs
        def capture(*a, **kw):
            result = original(*a, **kw)
            part = result[0][0, a[0].shape[1]:].tolist()[:limit-len(tokens)]
            if tok.eos_token_id in part:
                part = part[:part.index(tok.eos_token_id)+1]
            tokens.extend(part)
            lengths.append(len(part))
            raw_lengths.append(int(a[3])+1)
            return result
        with torch.inference_mode(), patch.object(upstream, 'update_inference_inputs', capture):
            stream = model.medusa_generate(ids, temperature=0.0, max_steps=limit, medusa_choices=choices)
            try:
                for _ in stream:
                    if len(tokens) >= limit or tok.eos_token_id in tokens:
                        break
            finally:
                stream.close()
        for attr in ['past_key_values','past_key_values_data','current_length_data']:
            if hasattr(model, attr):
                delattr(model, attr)
        out, stats = medusa_greedy_generate(model, ids, StaticTree(choices), max_new_tokens=limit)
        actual = out[0, ids.shape[1]:].tolist()
        cached, _ = plain_greedy_generate(model, ids, max_new_tokens=limit)
        from medusa.model.utils import reset_medusa_mode
        reset_medusa_mode(model)
        with torch.inference_mode():
            uncached = model.generate(ids, do_sample=False, max_new_tokens=limit,
                                      use_cache=False, pad_token_id=tok.eos_token_id)
        cached_tokens = cached[0, ids.shape[1]:].tolist()
        uncached_tokens = uncached[0, ids.shape[1]:].tolist()
        row = dict(prompt=prompt, tokens_match=actual == tokens,
                   acceptance_match=stats.accept_lengths == lengths,
                   raw_acceptance_match=stats.raw_accept_lengths == raw_lengths,
                   official_accept_lengths=lengths, custom_accept_lengths=stats.accept_lengths,
                   generated_ids=actual, text=tok.decode(actual, skip_special_tokens=True),
                   cached_matches_uncached=cached_tokens == uncached_tokens,
                   medusa_matches_cached=actual == cached_tokens,
                   medusa_matches_uncached=actual == uncached_tokens)
        rows.append(row)
        if len(saved_lengths) == len(PROMPTS):
            row['matches_saved_notebook_acceptance'] = stats.accept_lengths == saved_lengths[len(rows)-1]
        print(row, flush=True)
    out_path(cfg, 'results', 'official_validation.json').write_text(json.dumps(rows, indent=2))
    if not all(r['tokens_match'] and r['acceptance_match'] and r['raw_acceptance_match'] for r in rows):
        raise SystemExit('Official equivalence failed; inspect official_validation.json')
    print('PASS: official tokens and acceptance lengths match on all five notebook prompts.')

if __name__ == '__main__':
    main()
