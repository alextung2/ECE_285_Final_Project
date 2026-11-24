"""Compare actual verification decisions with cached-reference replay on a fixed prompt panel."""
import json
from pathlib import Path

import torch

from _common import parse
from mcal.data import vicuna_prompt
from mcal.decode import plain_greedy_generate, medusa_greedy_generate
from mcal.model import load_medusa
from mcal.policies import load_policy
from mcal.precision import require_compatible, trace_contract
from mcal.traces import TraceSet
from mcal.validation import TraceObserver


def extra(parser):
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True, help="Timing panel IDs, warmup first")


def main():
    args, cfg = parse(__doc__, extra)
    model, tok = load_medusa(cfg)
    folder = Path(cfg["out_dir"])
    contract = dict(model.execution_contract, n_heads=model.medusa, topk=cfg["traces"]["topk"])
    policies = {(b, name): load_policy(name, b, contract, folder / "models")
                for b in cfg["compare"]["budgets"] for name in cfg["speed"]["methods"]}
    records = json.loads(Path(cfg["traces"]["eval"]["prompt_file"]).read_text(encoding="utf-8"))
    by_id = {r["prompt_id"]: r["instruction"] for r in records}
    output = folder / "results" / "cached_verification.json"
    output.parent.mkdir(exist_ok=True, parents=True)
    results = []
    for entry in json.loads(args.panel.read_text())[1:]:
        pid = entry["question_id"]
        ids = tok([vicuna_prompt(by_id[pid])], return_tensors="pt").input_ids.to(model.device)
        reference, _ = plain_greedy_generate(model, ids, cfg["speed"]["max_new_tokens"])
        path = args.trace_dir / "eval" / str(pid)
        evidence = json.loads(path.with_suffix(".json").read_text())
        saved = evidence["generated_ids"]
        same_reference = saved == reference[0, ids.shape[1]:].tolist()
        ts = TraceSet.load(path.with_suffix(".npz")) if evidence["positions"] else None
        if ts is not None:
            require_compatible(contract, trace_contract(ts))
        for (b, name), policy in policies.items():
            observer = TraceObserver(ts, reference, ids.shape[1]) if ts is not None else None
            actual, _ = medusa_greedy_generate(model, ids, policy, cfg["speed"]["max_new_tokens"], observer=observer)
            results.append(dict(question_id=pid, category=entry["category"], budget=b, method=name,
                                saved_reference_agrees=same_reference, output_agrees=torch.equal(reference, actual),
                                rounds=observer.rows if observer is not None else []))
        output.write_text(json.dumps(results), encoding="utf-8")
        print("Verified panel question", pid, flush=True)


if __name__ == "__main__":
    main()
