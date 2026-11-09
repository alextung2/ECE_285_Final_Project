# Stage 1 FP32 trace samples

These are the calibration and evaluation smoke-test traces from the Colab run `ECE285_Medusa/fp32_v1`. Use them to review the trace format and exercise the loading and audit workflow. The CPU audit passed, but the strict GPU precision gate failed. Keep formal fitting, tree comparisons, and benchmarking pending until the gate passes.

## Files

| File | Contents |
| --- | --- |
| [`traces/calib.npz`](traces/calib.npz) | Calibration sample with 5 prompts and 295 labelled decoding positions. |
| [`traces/eval.npz`](traces/eval.npz) | Evaluation sample with 3 prompts and 177 labelled decoding positions. |
| [`audit_20261109-064020.json`](audit_20261109-064020.json) | Saved CPU audit, execution contracts, SHA-256 hashes, and provenance warnings. |
| [`prompts/calib_small.json`](prompts/calib_small.json) and [`prompts/eval_small.json`](prompts/eval_small.json) | Exact small prompt inputs supplied to the collector. |
| [`prompts/prompt_manifest.json`](prompts/prompt_manifest.json) | Prompt manifest from the Colab run. |
| [`logs/09_pilot_gate_20261109-051355.log`](logs/09_pilot_gate_20261109-051355.log) | Pilot precision gate. 15 of 20 prompts passed. The process exited with code 1. |
| [`logs/09_known_cases_20261109-040957.log`](logs/09_known_cases_20261109-040957.log) | Known-case precision gate. 6 of 7 prompts passed. The process exited with code 1. |

## Collection and validation

- Both samples were collected with `medusa-proj-skeleton/scripts/01_collect_traces.py`, `model.precision_mode=nf4_fp32_from_fp16`, `traces.max_new_tokens=64`, and `traces.keep_tokens=true`. The audit records NF4 weight storage and FP32 compute and Medusa heads.
- The CPU audit reports `offline_checks_passed: true`, `gpu_validation_performed: false`, and `label_truth_verified: false`. It checks structure and execution-contract compatibility. It also flags missing dataset revision metadata and calls for checking text overlap using source manifests because prompt IDs are from different or unknown datasets.
- The NPZ files are unchanged from Drive. Their SHA-256 hashes match the saved audit report.

To repeat the CPU audit, run the following from `medusa-proj-skeleton/` in the project Python environment.

```sh
python scripts/10_audit_traces.py \
  --calib ../shared_traces_fp32_stage1/traces/calib.npz \
  --eval ../shared_traces_fp32_stage1/traces/eval.npz \
  --report /tmp/fp32_stage1_audit.json \
  --set model.precision_mode=nf4_fp32_from_fp16
```

The [project precision guide](../medusa-proj-skeleton/PRECISION.md#commands-and-release-gate) describes the validation required before final collection.
