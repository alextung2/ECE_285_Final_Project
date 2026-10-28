"""Weeks 6-7: online end-to-end timing on MT-Bench (GPU required).

For each policy at budget B: tokens/sec, speedup vs ordinary greedy (1.0x),
online accepted tokens/step, tree-building overhead, and exact agreement with
greedy output. Only these numbers may be reported as measured speedups.

    python scripts/05_benchmark_speed.py
    python scripts/05_benchmark_speed.py --set speed.budget=32 speed.n_prompts=20
"""

import json

import numpy as np
from _common import parse, write_table


def main():
    args, cfg = parse(__doc__)
    if args.synthetic:
        raise SystemExit("05 needs the real model; there is no synthetic mode for wall-clock timing.")
    import torch
    from mcal.collect import default_medusa_choices
    from mcal.config import out_path
    from mcal.data import load_prompts, vicuna_prompt
    from mcal.decode import DynamicTree, StaticTree, medusa_greedy_generate, plain_greedy_generate
    from mcal.estimators import Scorer
    from mcal.model import load_medusa

    sc, B = cfg["speed"], cfg["speed"]["budget"]
    model, tok = load_medusa(cfg)
    ecfg = dict(cfg["traces"]["eval"], n_prompts=sc["n_prompts"] + sc["warmup_prompts"])
    prompts = load_prompts(ecfg)
    load = lambda n: Scorer.load(out_path(cfg, "models", f"{n}.pkl"))
    fixed = lambda n: json.loads(out_path(cfg, "models", f"fixed_tree_{n}_B{B}.json").read_text())

    policies = {
        "medusa_default": StaticTree(default_medusa_choices()),
        "medusa_acc[fixed]": StaticTree(fixed("medusa_acc")),
        "count_cond[fixed]": StaticTree(fixed("count_cond")),
        "medusa_plus": DynamicTree(load("medusa_plus"), B),
        "direct": DynamicTree(load("direct"), B),
        "logistic_cond": DynamicTree(load("logistic_cond"), B),
    }

    def encode(p):
        return tok([vicuna_prompt(p)], return_tensors="pt").input_ids

    # greedy reference
    ref_out, ref_tps = [], []
    for i, p in enumerate(prompts):
        out, st = plain_greedy_generate(model, encode(p), sc["max_new_tokens"])
        if i >= sc["warmup_prompts"]:
            ref_out.append(out.cpu()), ref_tps.append(st.tokens_per_s)
    base_tps = float(np.mean(ref_tps))
    rows = [dict(method="greedy", tokens_per_s=base_tps, speedup=1.0, accepted=1.0,
                 tree_ms=0.0, agree=1.0)]

    for name, pol in policies.items():
        tps, acc, tree_ms, agree = [], [], [], []
        for i, p in enumerate(prompts):
            out, st = medusa_greedy_generate(model, encode(p), pol, sc["max_new_tokens"])
            if i < sc["warmup_prompts"]:
                continue
            ref = ref_out[i - sc["warmup_prompts"]]
            n = min(ref.shape[1], out.shape[1])
            agree.append(float((ref[0, :n] == out[0, :n].cpu()).all()))
            tps.append(st.tokens_per_s), acc.append(st.mean_accepted)
            tree_ms.append(1e3 * st.tree_s / max(st.steps, 1))
        rows.append(dict(method=name, tokens_per_s=float(np.mean(tps)),
                         speedup=float(np.mean(tps)) / base_tps, accepted=float(np.mean(acc)),
                         tree_ms=float(np.mean(tree_ms)), agree=float(np.mean(agree))))
        print(rows[-1])
        torch.cuda.empty_cache()

    print(write_table(rows, out_path(cfg, "results", f"speed_B{B}.csv")))


if __name__ == "__main__":
    main()
