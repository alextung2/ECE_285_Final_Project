"""Inspect first dynamic/greedy divergence. Diagnostic instrumentation is not timing."""
import json
from unittest.mock import patch
from _common import parse


def main():
    _, cfg = parse(__doc__)
    import torch
    import medusa.model.utils as utils
    from mcal.model import load_medusa
    from mcal.decode import medusa_greedy_generate, plain_greedy_generate, DynamicTree
    from mcal.estimators import MedusaPlusProb
    from mcal.data import vicuna_prompt
    from mcal.config import out_path
    model, tok = load_medusa(cfg)
    budget = cfg.get('diagnose', {}).get('budget', 8)
    prompt = 'Write a short story about a lost key.'
    ids = tok([vicuna_prompt(prompt)], return_tensors='pt').input_ids.to(model.device)
    def top2(logits):
        values, tokens = logits.float().topk(2)
        return dict(tokens=tokens.tolist(), logits=values.tolist(), gap=float(values[0]-values[1]),
                    argmax_token=int(logits.argmax()))
    greedy_scores = []
    forward = model.forward
    def trace_forward(*a, **kw):
        out = forward(*a, **kw)
        greedy_scores.append(top2(out.logits[0,-1]))
        return out
    with patch.object(model, 'forward', trace_forward):
        greedy, _ = plain_greedy_generate(model, ids, 64)
    dynamic_scores, state = [], {}
    initialize, update = utils.initialize_medusa, utils.update_inference_inputs
    def trace_init(*a, **kw):
        result = initialize(*a, **kw)
        state['root'] = result[1][0,-1]
        return result
    def trace_update(*a, **kw):
        best, acc, logits = a[2], int(a[3]), a[6]
        dynamic_scores.append(top2(state['root']))
        dynamic_scores.extend(top2(logits[best,j-1]) for j in range(1,acc+1))
        result = update(*a, **kw)
        state['root'] = result[1][0,-1]
        return result
    with patch.object(utils, 'initialize_medusa', trace_init), patch.object(utils, 'update_inference_inputs', trace_update):
        dynamic, _ = medusa_greedy_generate(model, ids, DynamicTree(MedusaPlusProb(),budget), 64)
    a, b = greedy[0,ids.shape[1]:].tolist(), dynamic[0,ids.shape[1]:].tolist()
    first = next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None)
    report = dict(prompt=prompt, budget=budget, exact_agreement=a==b,
                  first_different_position=first, greedy_ids=a, dynamic_ids=b)
    if first is not None:
        report.update(greedy_prediction=greedy_scores[first], dynamic_prediction=dynamic_scores[first],
                      common_prefix=tok.decode(a[:first]),
                      greedy_token=tok.decode([a[first]]), dynamic_token=tok.decode([b[first]]))
    path = out_path(cfg, 'results', f'dynamic_divergence_B{budget}.json')
    path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__ == '__main__':
    main()
