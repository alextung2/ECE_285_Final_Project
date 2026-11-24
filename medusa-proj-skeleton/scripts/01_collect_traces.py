"""Week 2: collect verification traces (calibration = AlpacaEval, evaluation = MT-Bench).

    python scripts/01_collect_traces.py --split calib
    python scripts/01_collect_traces.py --split eval
    python scripts/01_collect_traces.py --synthetic          # no GPU: writes synthetic traces
"""

import time
import json

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
    from mcal.provenance import read_prompt_records, prompt_digest

    model, tok = load_medusa(cfg)
    for split in splits:
        if tcfg[split].get("prompt_file"):
            prompt_records = read_prompt_records(tcfg[split]["prompt_file"])
            prompts = [r["instruction"] for r in prompt_records]
            prompt_ids = [r["prompt_id"] for r in prompt_records]
        else:
            prompts = load_prompts(tcfg[split])
            prompt_ids = list(range(len(prompts)))
            prompt_records = [dict(prompt_id=i, instruction=p) for i, p in zip(prompt_ids, prompts)]
        digest = prompt_digest(prompt_records)
        parts, agree, t0 = [], [], time.time()
        dest = out_path(cfg, tcfg[split]["file"])
        if dest.exists():
            raise FileExistsError(f"Preserve previous traces; choose a new out_dir: {dest}")
        if dest.with_suffix('.failure.json').exists() or dest.with_suffix('.prompts.json').exists():
            raise FileExistsError(f"Previous attempt exists; choose a new out_dir: {dest}")
        dest.with_suffix('.prompts.json').write_text(json.dumps(prompt_records, indent=2), encoding="utf-8")
        for i, p in enumerate(prompts):
            try:
                ts, a = traces_for_prompt(model, tok, p, prompt_ids[i], tcfg["topk"], tcfg["max_new_tokens"],
                                          tcfg.get("keep_tokens", False))
            except (ValueError, RuntimeError) as e:
                failure = dict(prompt_id=prompt_ids[i], prompt=p, error=str(e),
                               execution_contract=model.execution_contract, collection_complete=False,
                               expected_prompt_ids=prompt_ids, completed_prompt_ids=prompt_ids[:i],
                               prompt_sha256=digest)
                dest.with_suffix('.failure.json').write_text(json.dumps(failure, indent=2))
                raise
            ts.meta.update(split=split, dataset=tcfg[split]["dataset"],
                           dataset_revision=tcfg[split].get("dataset_revision"),
                           prompt_sha256=digest, expected_prompt_ids=prompt_ids,
                           collection_complete=False)
            parts.append(ts), agree.append(a)
            if (i + 1) % 20 == 0:
                print(f"[{split}] {i + 1}/{len(prompts)}  {time.time() - t0:.0f}s  "
                      f"greedy agreement {np.mean(agree):.4f}")
                TraceSet.concat(parts).save(dest)      # checkpoint
        ts = TraceSet.concat(parts)
        ts.meta.update(model=cfg["model"]["name"], split=split, dataset=tcfg[split]["dataset"],
                       load_in_4bit=cfg["model"].get("load_in_4bit", False),
                       greedy_agreement=float(np.mean(agree)))
        ts.meta['collection_complete'] = True
        ts.save(dest)
        print(split, ts.summary())


if __name__ == "__main__":
    main()
