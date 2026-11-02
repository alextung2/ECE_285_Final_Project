"""Online timing. Defaults to default Medusa and MEDUSA+, without fitted models.

Use --set speed.methods=[medusa_default,medusa_plus,direct,logistic_cond,count_cond]
after the fitted model handoff. Outputs include per-prompt tokens, time, exact
agreement, environment metadata, and separately synchronized overhead profiles.
"""
import json
import platform
import importlib.metadata
import hashlib
from dataclasses import asdict
from pathlib import Path
import numpy as np
from _common import parse, write_table


def exact_agreement(a, b):
    return a.shape == b.shape and bool((a.cpu() == b.cpu()).all())


def main(model_and_tokenizer=None):
    args, cfg = parse(__doc__)
    if args.synthetic:
        raise SystemExit('Timing requires a real GPU model; synthetic speedups are not supported.')
    import torch
    from mcal.collect import default_medusa_choices
    from mcal.config import out_path
    from mcal.data import load_prompts, vicuna_prompt
    from mcal.decode import DynamicTree, StaticTree, medusa_greedy_generate, plain_greedy_generate
    from mcal.estimators import Scorer, MedusaPlusProb
    from mcal.model import load_medusa

    sc, B = cfg['speed'], cfg['speed']['budget']
    methods = sc.get('methods', ['medusa_default', 'medusa_plus'])
    if sc['n_prompts'] < 1 or sc['warmup_prompts'] < 1 or sc['max_new_tokens'] < 1:
        raise ValueError('Use at least one warmup, measured prompt, and generated token.')
    from mcal.precision import require_compatible, precision_spec
    model, tok = model_and_tokenizer or load_medusa(cfg)
    expected = precision_spec(cfg["model"])
    if any(model.execution_contract.get(k) != v for k, v in expected.items()):
        raise ValueError("Reused model differs from requested precision")
    contract = dict(model.execution_contract, n_heads=model.medusa, topk=cfg['traces']['topk'])
    policies = {}
    for name in methods:
        if name == 'medusa_default':
            policies[name] = StaticTree(default_medusa_choices())
        elif name == 'medusa_plus':
            policies[name] = DynamicTree(MedusaPlusProb(), B)
        elif name in ('medusa_acc', 'count_cond'):
            path = out_path(cfg, 'models', f'fixed_tree_{name}_B{B}.json')
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                raise ValueError(f"Legacy tree lacks execution contract: {path}")
            require_compatible(contract, payload.get('execution_contract'))
            policies[name] = StaticTree(payload['choices'])
            if len(policies[name].choices) != B:
                raise ValueError(f'{path} does not contain {B} draft nodes')
        elif name in ('direct', 'logistic_cond'):
            scorer = Scorer.load(out_path(cfg, 'models', f'{name}.pkl'))
            require_compatible(contract, getattr(scorer, 'execution_contract', None))
            policies[name] = DynamicTree(scorer, B)
        else:
            raise ValueError(f'Unknown policy: {name}')

    if sc.get('prompt_file'):
        prompts = json.loads(Path(sc['prompt_file']).read_text(encoding='utf-8-sig'))
    else:
        ecfg = dict(cfg['traces']['eval'], n_prompts=sc['n_prompts'] + sc['warmup_prompts'])
        if 'path' in ecfg and not Path(ecfg['path']).is_absolute():
            ecfg['path'] = str(Path(__file__).resolve().parents[1] / ecfg['path'])
        prompts = load_prompts(ecfg)
    needed = sc['n_prompts'] + sc['warmup_prompts']
    if len(prompts) < needed:
        raise ValueError(f'Need {needed} prompts, found {len(prompts)}')
    prompts = prompts[:needed]
    encode = lambda p: tok([vicuna_prompt(p)], return_tensors='pt').input_ids
    records = []
    rng = np.random.default_rng(cfg['seed'])
    # Pair all policies on each prompt; randomize policy order to reduce thermal/order bias.
    for i, prompt in enumerate(prompts):
        ids = encode(prompt)
        outputs, stats = {}, {}
        names = ['greedy'] + list(policies)
        rng.shuffle(names)
        for name in names:
            out, st = (plain_greedy_generate(model, ids, sc['max_new_tokens']) if name == 'greedy'
                       else medusa_greedy_generate(model, ids, policies[name], sc['max_new_tokens']))
            outputs[name], stats[name] = out.cpu(), st
        if i < sc['warmup_prompts']:
            continue
        for name in names:
            st = stats[name]
            record = dict(precision_mode=expected["precision_mode"], prompt_id=i-sc['warmup_prompts'], method=name,
                          draft_budget=(0 if name == 'greedy' else len(policies[name].choices)
                                        if not policies[name].dynamic else B),
                          agree=exact_agreement(outputs[name], outputs['greedy']),
                          generated_ids=outputs[name][0, ids.shape[1]:].tolist(), **asdict(st))
            records.append(record)
        print(f'Completed measured prompt {i-sc["warmup_prompts"]+1}/{sc["n_prompts"]}', flush=True)
        out_path(cfg, 'results', f'speed_B{B}_prompts.json').write_text(json.dumps(records, indent=2))

    # Synchronization changes runtime; profile overhead in separate runs, never use
    # those wall times for speedup. Profiling is bounded to the first measured prompt.
    overhead = {}
    for name, policy in policies.items():
        _, st = medusa_greedy_generate(model, encode(prompts[sc['warmup_prompts']]), policy,
                                      sc['max_new_tokens'], profile_tree=True)
        overhead[name] = 1000 * st.tree_s / max(st.steps, 1)
    baseline = [r for r in records if r['method'] == 'greedy']
    base_tps = sum(r['new_tokens'] for r in baseline) / sum(r['wall_s'] for r in baseline)
    rows = []
    for name in ['greedy'] + list(policies):
        rr = [r for r in records if r['method'] == name]
        tokens, seconds = sum(r['new_tokens'] for r in rr), sum(r['wall_s'] for r in rr)
        rows.append(dict(precision_mode=expected["precision_mode"], method=name, draft_budget=rr[0]['draft_budget'], tokens_per_s=tokens/seconds,
                         speedup=(tokens/seconds)/base_tps,
                         accepted=tokens/sum(r['steps'] for r in rr),
                         tree_ms_profile=overhead.get(name, 0), agree=np.mean([r['agree'] for r in rr])))
    print(write_table(rows, out_path(cfg, 'results', f'speed_B{B}.csv')))
    packages = {p: importlib.metadata.version(p) for p in ['torch','transformers','bitsandbytes','numpy']}
    metadata = dict(execution_contract=model.execution_contract, precision_audit=model.precision_audit, config=cfg, packages=packages, platform=platform.platform(),
                    gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                    prompt_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
                    timing='Prefill + generation; excludes model load, tokenization, KV allocation and fixed-buffer setup.',
                    overhead='Separate synchronized run on first measured prompt; includes policy and buffer lookup/build.',
                    note='Hardware-specific preliminary results. Default tree retains its native budget; inspect draft_budget.')
    out_path(cfg, 'results', f'speed_B{B}_metadata.json').write_text(json.dumps(metadata, indent=2))

if __name__ == '__main__':
    main()
