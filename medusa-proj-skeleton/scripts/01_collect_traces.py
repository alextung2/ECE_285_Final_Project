"""Week 2: collect verification traces (calibration = AlpacaEval, evaluation = MT-Bench).

    python scripts/01_collect_traces.py --split calib
    python scripts/01_collect_traces.py --split eval
    python scripts/01_collect_traces.py --synthetic          # no GPU: writes synthetic traces
"""

import time

import numpy as np
from _common import parse


def extra(ap):
    ap.add_argument("--split", choices=["calib", "eval", "both"], default="both")


def main():
    args, cfg = parse(__doc__, extra)
    from mcal.config import out_path
    from mcal.traces import TraceSet, make_synthetic_traces

    splits = ["calib", "eval"] if args.split == "both" else [args.split]
    tcfg = cfg["traces"]

    if args.synthetic:
        s = tcfg["synthetic"]
        for i, split in enumerate(splits):
            n = s["n_prompts_calib"] if split == "calib" else s["n_prompts_eval"]
            ts = make_synthetic_traces(n, s["steps_per_prompt"], s["n_heads"], tcfg["topk"],
                                       s["dependence"], seed=cfg["seed"] + 100 * (split == "eval"))
            ts.save(out_path(cfg, tcfg[split]["file"]))
            print(split, ts.summary())
        return

    from mcal.collect import traces_for_prompt
    from mcal.data import load_prompts
    from mcal.model import load_medusa

    model, tok = load_medusa(cfg)
    for split in splits:
        prompts = load_prompts(tcfg[split])
        parts, agree, t0 = [], [], time.time()
        dest = out_path(cfg, tcfg[split]["file"])
        for i, p in enumerate(prompts):
            try:
                ts, a = traces_for_prompt(model, tok, p, i, tcfg["topk"], tcfg["max_new_tokens"],
                                          tcfg.get("keep_tokens", False))
            except ValueError as e:
                print(f"skip prompt {i}: {e}")
                continue
            parts.append(ts), agree.append(a)
            if (i + 1) % 20 == 0:
                print(f"[{split}] {i + 1}/{len(prompts)}  {time.time() - t0:.0f}s  "
                      f"greedy agreement {np.mean(agree):.4f}")
                TraceSet.concat(parts).save(dest)      # checkpoint
        ts = TraceSet.concat(parts)
        ts.meta.update(model=cfg["model"]["name"], split=split, dataset=tcfg[split]["dataset"],
                       load_in_4bit=cfg["model"].get("load_in_4bit", False),
                       greedy_agreement=float(np.mean(agree)))
        ts.save(dest)
        print(split, ts.summary())
        if np.mean(agree) < 0.995:
            print("WARNING: teacher-forced argmax disagrees with generated tokens; labels may be off.")


if __name__ == "__main__":
    main()
