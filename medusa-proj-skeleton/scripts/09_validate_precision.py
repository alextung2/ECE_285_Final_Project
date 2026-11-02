"""Correctness only: cached/uncached/static/dynamic and offline-online labels.

Run once per precision mode in a NEW process. Use --prompt-file for Lucia's
pilot/train/validation JSON or a list of strings. No latency claim is emitted.
"""
import json
from pathlib import Path
from _common import parse

PROMPTS = [
    'Explain what a robot is in one sentence.',
    'What is the difference between weather and climate?',
    'Explain why the sky appears blue.',
    "Translate 'Good morning, how are you?' into French.",
    'Write a Python function that returns the square of a number.',
    'Write a short story about a lost key.',
    'What breed dog is smallest?',
]


def extra(ap):
    ap.add_argument('--prompt-file', type=Path)


def main():
    args, cfg = parse(__doc__, extra)
    if args.synthetic:
        raise SystemExit('Requires GPU/model weights; synthetic results cannot certify precision')
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: real precision validation NOT RUN')
    from mcal.collect import traces_for_prompt, default_medusa_choices
    from mcal.config import out_path
    from mcal.data import vicuna_prompt
    from mcal.decode import plain_greedy_generate, medusa_greedy_generate, StaticTree, DynamicTree
    from mcal.estimators import MedusaPlusProb
    from mcal.model import load_medusa
    from mcal.validation import TraceObserver
    from medusa.model.utils import reset_medusa_mode
    prompts = PROMPTS
    if args.prompt_file:
        values = json.loads(args.prompt_file.read_text(encoding='utf-8'))
        prompts = [r['instruction'] if isinstance(r, dict) else r for r in values]
    model, tok = load_medusa(cfg)
    limit = cfg['speed']['max_new_tokens']
    report = dict(execution_contract=model.execution_contract, audit=model.precision_audit,
                  timing_benchmark=False, prompts=[], passed=False)
    dest = out_path(cfg, 'results', 'precision_validation.json')
    if dest.exists():
        raise FileExistsError('Use a new out_dir; preserve previous validation evidence')
    for i, prompt in enumerate(prompts):
        row = dict(prompt=prompt, policies=[], passed=False)
        report['prompts'].append(row)
        try:
            torch.cuda.reset_peak_memory_stats()
            ids = tok([vicuna_prompt(prompt)], return_tensors='pt').input_ids.to(model.device)
            reference, _ = plain_greedy_generate(model, ids, limit)
            reset_medusa_mode(model)
            with torch.inference_mode():
                uncached = model.generate(ids, do_sample=False, max_new_tokens=limit,
                                          use_cache=False, pad_token_id=tok.eos_token_id)
            row['cached_ids'] = reference[0, ids.shape[1]:].tolist()
            row['uncached_ids'] = uncached[0, ids.shape[1]:].tolist()
            row['cached_uncached_agree'] = torch.equal(reference, uncached)
            ts, _ = traces_for_prompt(model, tok, prompt, i, max_new_tokens=limit, keep_tokens=True)
            policies = [('static', StaticTree(default_medusa_choices()))] + [
                (f'dynamic_B{b}', DynamicTree(MedusaPlusProb(), b)) for b in cfg['compare']['budgets']]
            for name, policy in policies:
                observer = TraceObserver(ts, reference, ids.shape[1])
                output, _ = medusa_greedy_generate(model, ids, policy, limit, observer=observer)
                row['policies'].append(dict(name=name, exact_agreement=torch.equal(output, reference),
                                           generated_ids=output[0, ids.shape[1]:].tolist(),
                                           offline_online_agree=observer.passed, rounds=observer.rows))
            row['passed'] = row['cached_uncached_agree'] and all(
                p['exact_agreement'] and p['offline_online_agree'] for p in row['policies'])
            row['diagnostic_peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
        except (ValueError, RuntimeError) as exc:
            row['error'] = str(exc)
        dest.write_text(json.dumps(report, indent=2))
        print(f'Prompt {i}: passed={row["passed"]}', flush=True)
    report['passed'] = bool(report['prompts']) and all(r['passed'] for r in report['prompts'])
    dest.write_text(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(f'Precision gate FAILED; inspect {dest}')


if __name__ == '__main__':
    main()
