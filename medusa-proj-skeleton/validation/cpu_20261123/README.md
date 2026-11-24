# CPU readiness checks, 2026-11-23

Base: main `98c34c2084e821541749aa0f5e7f6da66669b76e`, in an isolated worktree.
The first regression run found two broken references to Lucia's moved precision
scripts: 67 passed, 2 failed, 2 skipped. After repairing the paths and adding
coverage for the changes here, the full suite reports **83 passed, 2 skipped**.
See [pytest.txt](pytest.txt) for the executed result and skip reasons.

This laptop exposes Intel Iris Xe graphics; PyTorch reports CUDA unavailable.
All work here used CPU execution. Medusa/FastChat integration tests were skipped
because those packages are not installed. Controlled torch CPU fixtures do not
exercise bitsandbytes CUDA kernels.

## Checks completed

- Full regression suite, Python compilation, and Git whitespace checks.
- Both archived Lucia promotion implementations match the canonical promotion
  operations on CPU fixtures, now using the reorganized source paths.
- Stage 1 NPZ structure, execution-contract agreement, original SHA-256 hashes,
  exact ordered prompt ID/text hashes, and complete 5/3 sample prompt coverage.
  The supplied sample prompt sets have no normalized text overlap.
- Both samples produce finite conditional and candidate-pool features. MEDUSA+
  constructs valid trees at budgets 8, 16, 32, 63, 96. No final model was fitted
  on those samples and their label truth remains unverified.
- CLI synthetic stages 01, 02, 03, 04: 400 calibration and 160 evaluation
  positions, 40 pool steps, self-retraining enabled, 20 compared evaluation
  steps, all five methods and budgets.
- Every code cell of Ada's integrated notebook: synthetic 20x20 calibration,
  8x20 evaluation, POOL_STEPS=40 and QUICK_EVAL_STEPS=20; tuning, direct-pool
  investigation, grouped ablations, export and score round-trip checks completed.
  All 25 method/budget comparisons also completed using the notebook exports.
  [notebook_smoke.txt](notebook_smoke.txt) records execution. The committed
  notebook and its historical counterpart were not modified.
- Experiment preparation completed with both a verified public MT-Bench download
  and a local copy of the same bytes, producing identical prompt hashes.

## New issue found and corrected before collection

Original training prompt 801 exactly duplicates MT-Bench question 81. The new
preparation script removes that prompt from training inputs only and writes
an exclusion record plus the original manifest. It prepares 199 training,
50 validation and 80 evaluation prompts. No recorded trace, original prompt
file or failed case was removed. The historical pilot still includes prompt 801.
This is leakage prevention before fitting, not an exclusion based on outcomes.

## Evidence files

- [stage1_prompt_audit.json](stage1_prompt_audit.json): read-only sample audit.
- [sample_interoperability.json](sample_interoperability.json): feature/tree checks.
- [preparation.json](preparation.json): input counts, hashes, explicit exclusion.
- [summary.json](summary.json): CPU environment and synthetic integration sizes.
- `cli_synthetic_tree_comparison.csv` and
  `notebook_synthetic_tree_comparison.csv`: labeled pipeline fixtures only.
  They are **not** final results, evidence of a better method, or GPU speedups.

To reproduce the sample audit from `medusa-proj-skeleton/`, choose a new report:

```sh
python scripts/10_audit_traces.py --calib ../shared_traces_fp32_stage1/traces/calib.npz --eval ../shared_traces_fp32_stage1/traces/eval.npz --calib-prompts ../shared_traces_fp32_stage1/prompts/calib_small.json --eval-prompts ../shared_traces_fp32_stage1/prompts/eval_small.json --report outputs/stage1_prompt_audit.json --set model.precision_mode=nf4_fp32_from_fp16
```

## Still unverified

No real model was loaded; no CUDA correctness, complete collection, real-data
tuning, trained-policy verification, GPU memory or latency experiment was run.
The committed GPU failures (15/20 pilot, 6/7 known cases passing) remain unresolved.
The existing Stage 1 metadata lacks dataset revision and an explicit completion
flag; external prompt/hash checks verify coverage only, not acceptance labels.
Future collection records those fields and stops instead of silently skipping
unlabelable prompts. See [the next-stage workflow](../../FINAL_EXPERIMENTS.md).
