# Final experiment workflow

Status on 2026-11-23: CPU integration is ready for a GPU investigation, not a
completed experiment. See [the executed CPU checks](validation/cpu_20261123/README.md).
NF4 storage with `nf4_fp32_from_fp16` execution remains provisional:
the committed Stage 1 correctness checks passed 15/20 pilot and 6/7 known prompts.
The detailed `precision_validation.json` files are not committed; the text logs
alone cannot distinguish candidate-rank, label, teacher-forced, or output failures.
Obtain those JSON reports from the original run or reproduce them on DataHub.
Do not remove failing prompts or relax exact-agreement checks.

## Prepared inputs and the leakage correction

Run from `medusa-proj-skeleton/` on the machine that will perform the experiment:

```sh
python scripts/11_prepare_experiment.py --out outputs/fp32_run01 --max-new-tokens 64
```

This CPU-only command freezes prompt files, their hashes, an explicit FP32
configuration, a train/validation manifest, and a preparation report. It
downloads the public FastChat v0.2.36 MT-Bench file and verifies its bytes against
Lucia's recorded SHA-256; `--mtbench-file` accepts a local verified copy.
It refuses an existing destination. Generated paths are absolute: regenerate
on DataHub instead of copying a Windows config there.

The original training prompt 801 is identical to MT-Bench question 81 (Hawaii
travel blog). Preparation excludes it from the **new training input**, leaving
199 training, 50 validation and 80 evaluation prompts. Both the original
manifest and an explicit exclusion record are saved. All original input files
and traces stay unchanged. Prompt 801 remains in the historical pilot used for
correctness diagnosis. No exact or Unicode/case/whitespace-normalized overlap
remains; this does not rule out semantic similarity.

The 64-token setting matches the current pilot horizon and is a bounded starting
protocol, not evidence about longer generations. Choose any longer horizon
before collection and use it consistently in validation and benchmarking.
The generated default benchmark uses 2 warmup and 40 measured evaluation prompts;
state that subset in the report. The five matched draft budgets are 8, 16, 32,
63 and 96. Default Medusa's native tree is a separate reference, not a
budget-matched policy at every B.

## GPU stage: correctness before collection

Check GPU allocation, free VRAM, disk quota and weight-cache space first. Use a
fresh Python 3.11 environment with the pinned versions in
`requirements-alex.txt` and Medusa commit
`e2a5d20c048a9b0a4092e6933c34313687422518`. Select the CUDA torch build only
after checking DataHub's driver. The Stage 1 run used torch 2.2.2+cu121;
different runtime contracts require new traces and fitted artifacts.

For a Linux shell, after preparing the directory:

```sh
RUN="$(pwd)/outputs/fp32_run01"
CFG="$RUN/config.yaml"
python scripts/00_gpu_check.py --config "$CFG"
python scripts/06_validate_official.py --config "$CFG" --set out_dir="$RUN/official"
python scripts/09_validate_precision.py --config "$CFG" --set out_dir="$RUN/known"
python scripts/09_validate_precision.py --config "$CFG" --prompt-file ../medusa_trace_collection/prompts/pilot_prompts.json --set out_dir="$RUN/pilot"
```

These are individual stages, not a script to run through despite errors.
Inspect every report and require a zero exit status and exact agreement before
continuing. A failed report is evidence to retain. FP32 does not guarantee
shape-independent argmax decisions. No GPU command above was executed in the
CPU validation session.

## Collection and CPU analysis after the gate passes

```sh
python scripts/01_collect_traces.py --config "$CFG"
python scripts/10_audit_traces.py --config "$CFG" --calib "$RUN/traces/calib.npz" --eval "$RUN/traces/eval.npz" --calib-prompts "$RUN/prompts/calib_full.json" --eval-prompts "$RUN/prompts/eval_full.json" --report "$RUN/trace_audit.json"
python scripts/02_diagnose.py --config "$CFG"
```

The collector now stops on an unlabelable short response as well as numerical
disagreement, recording the offending prompt. Periodic NPZ checkpoints are
marked incomplete and rejected by audits. A completed run records exact expected
IDs, source revision, and a copy/hash of the actual ordered prompt inputs.
Do not invent labels for short responses; investigate and document any protocol
change before a fresh run. The historical Colab notebook uses the old prompt
split/chunk workflow: use the preparation command above for this new experiment.

For Ada's full tuning/ablation workflow, use
`Step 2 Calibration Integrated.ipynb` with a fresh kernel and:

- `USE_SYNTHETIC=False`, `PRECISION_MODE="nf4_fp32_from_fp16"`.
- `CALIB_FILE=RUN/"traces/calib.npz"`, `EVAL_FILE=RUN/"traces/eval.npz"`.
- `MANIFEST_FILE=RUN/"prompts/split_manifest.json"`.
- A fresh `OUT_DIR`. Tune on the validation split only; do not select settings
  from the evaluation results. Preserve the settings, ablations and trace hashes.

The simpler `03_fit_estimators.py --config "$CFG"` is also executable but uses
configured hyperparameters; it does **not** replace Ada's validation search.
Both paths were exercised on synthetic inputs during CPU validation.
For notebook exports, point `out_dir` at the notebook's `real` output folder
and `estimators.manifest` at its exported `split_manifest.json`:

```sh
python scripts/04_compare_trees.py --config "$CFG" --set out_dir=/path/to/notebook/real estimators.manifest=/path/to/notebook/real/split_manifest.json
```

Never fit the current Stage 1 samples as final research data: their broader
GPU gate failed and their 5/3 prompts are only interoperability samples.

## Trained-policy correctness, then separate latency

`09_validate_precision.py --trained-model-dir /path/to/models` adds all five
trained policies at every configured budget. It uses the same artifact loading
and contract checks as the benchmark, checks offline/online ranks and labels,
and records artifact hashes. Run it on the known cases and evaluation prompts
in new output directories. Keep the seven known cases and all pilot cases even
when a case fails. Only after the trained policies pass should latency runs start.

```sh
python scripts/09_validate_precision.py --config "$CFG" --trained-model-dir /path/to/notebook/real/models --prompt-file "$RUN/prompts/eval_full.json" --set out_dir="$RUN/trained_correctness"
python scripts/07_benchmark_budgets.py --config "$CFG" --set out_dir=/path/to/notebook/real
```

The benchmark includes matched-precision cached greedy, randomizes method order,
and profiles overhead in separate synchronized runs. Observers belong only in
the correctness stage. Retain per-prompt outputs and agreement columns even if
timing disagrees with the earlier gate. Repeats must use fresh result directories
with copies of the same fitted artifacts; never overwrite an earlier timing run.

## Report completion criteria

Keep these evidence types distinct:

| Claim | Required artifacts |
| --- | --- |
| Precision and label consistency | Runtime contract, all correctness JSON reports, exact agreement and failures |
| Calibration quality | Real trace hashes, split/exclusion manifest, Brier/ECE/reliability, tuning/ablation settings |
| Better trees at matched budget | Real trace comparison, paired prompt bootstrap CIs, methods and budgets |
| Actual decoding speedup | Same-device/precision greedy baseline, per-prompt timing, agreement, GPU/runtime/memory |

Report sample sizes, exclusions, token horizon, hardware, and limitations:
finite prompt coverage, dependence on numerical execution, offline visitation
bias, and possible calibration gains without end-to-end speedup. Fill final
tables and conclusions only after those artifacts exist. The submitted midterm
PDF is a historical snapshot and should remain unchanged.
