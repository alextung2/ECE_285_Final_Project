"""CPU-only: validate and freeze full prompt inputs/config for a new GPU run.

Downloads only the public MT-Bench JSONL, verified against the committed hash.
Does not load weights, collect traces, fit models, or certify precision.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import yaml
from _common import ROOT
from mcal.config import load_config
from mcal.provenance import read_prompt_records, require_disjoint_text, exclude_evaluation_overlap


def prepare(destination, mtbench_file=None, max_new_tokens=64):
    dest = Path(destination).resolve()
    if dest.exists():
        raise FileExistsError("Choose a new run directory; previous artifacts are preserved")
    if max_new_tokens <= 5:
        raise ValueError("Token horizon must exceed the five Medusa heads")
    repo = ROOT.parent
    source = repo / "medusa_trace_collection/prompts"
    manifest = json.loads((source / "data_manifest.json").read_text())
    train = read_prompt_records(source / "train_prompts.json")
    validation = read_prompt_records(source / "validation_prompts.json")
    for name, records in (("train", train), ("validation", validation)):
        if [r["prompt_id"] for r in records] != manifest[name + "_prompt_ids"]:
            raise ValueError(f"{name} prompt IDs differ from original manifest")
    if set(manifest["train_prompt_ids"]) & set(manifest["validation_prompt_ids"]):
        raise ValueError("Training/validation ID overlap")
    require_disjoint_text(train, validation)
    reference = json.loads((repo / "shared_traces_fp32_stage1/prompts/prompt_manifest.json").read_text())
    url = reference["eval"]["source"]
    if mtbench_file:
        raw = Path(mtbench_file).read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != reference["sha256"]["mt_bench_question_fastchat_v0.2.36.jsonl"]:
        raise ValueError("MT-Bench bytes differ from Lucia's recorded source hash")
    questions = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    evaluation = [dict(prompt_id=q["question_id"], instruction=q["turns"][0]) for q in questions]
    if [r["prompt_id"] for r in evaluation] != reference["eval"]["prompt_ids"]:
        raise ValueError("MT-Bench question IDs differ from source manifest")
    original_manifest = manifest
    train, train_exclusions = exclude_evaluation_overlap(train, evaluation)
    validation, validation_exclusions = exclude_evaluation_overlap(validation, evaluation)
    exclusions = [dict(split="train", **r) for r in train_exclusions]
    exclusions += [dict(split="validation", **r) for r in validation_exclusions]
    manifest = dict(manifest, train_prompt_ids=[r["prompt_id"] for r in train],
                    validation_prompt_ids=[r["prompt_id"] for r in validation],
                    exclusions=exclusions)
    overlap = require_disjoint_text(train + validation, evaluation)
    inputs = dest / "prompts"
    inputs.mkdir(parents=True)
    files = {"calib_full.json": train + validation, "eval_full.json": evaluation,
             "timing_prompts.json": [r["instruction"] for r in evaluation],
             "split_manifest.json": manifest, "original_split_manifest.json": original_manifest}
    for name, data in files.items():
        (inputs / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (inputs / "mt_bench_question.jsonl").write_bytes(raw)
    cfg = load_config(ROOT / "configs/default.yaml")
    cfg["out_dir"] = str(dest)
    cfg["model"]["precision_mode"] = "nf4_fp32_from_fp16"
    cfg["traces"].update(keep_tokens=True, max_new_tokens=max_new_tokens)
    for split, dataset, revision, count in [
        ("calib", manifest["dataset"], manifest["revision"], len(train + validation)),
        ("eval", "mtbench", "sha256:" + sha, len(evaluation)),
    ]:
        cfg["traces"][split].update(dataset=dataset, dataset_revision=revision, n_prompts=count,
                                   prompt_file=str(inputs / (split + "_full.json")))
    cfg["estimators"]["manifest"] = str(inputs / "split_manifest.json")
    cfg["speed"].update(max_new_tokens=max_new_tokens,
                        prompt_file=str(inputs / "timing_prompts.json"),
                        methods=["medusa_acc", "medusa_plus", "count_cond", "logistic_cond", "direct"])
    (dest / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    status = dict(prepared_only=True, gpu_validation_performed=False, precision_gate_passed=False,
                  train_prompts=len(train), validation_prompts=len(validation), eval_prompts=len(evaluation),
                  exclusions=exclusions,
                  text_overlap_check=overlap, mtbench_source=url, max_new_tokens=max_new_tokens,
                  sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs.iterdir())})
    (dest / "preparation.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mtbench-file", type=Path, help="Use a local copy instead of downloading")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()
    print(json.dumps(prepare(args.out, args.mtbench_file, args.max_new_tokens), indent=2))
