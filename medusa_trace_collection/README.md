# Medusa trace collection archive

Historical inputs, scripts, diagnostics, and pilot outputs for the Medusa trace collection runs.

| Directory | Contents |
| --- | --- |
| [`collectors/`](collectors/) | FP32 pilot and FP16 shared trace collection scripts. |
| [`precision/`](precision/) | FP32 candidate loader, precision test, mismatch diagnostic, and saved prompt 24 precision report. |
| [`prompts/`](prompts/) | Pilot, training, and validation prompt JSON files and the dataset manifest. |
| [`environment/`](environment/) | Environment snapshot from the original pilot. |
| [`pilot_results/fp32/`](pilot_results/fp32/) | Historical FP32 per-prompt node traces, collection settings, and summary. |
| [`pilot_results/fp16/`](pilot_results/fp16/) | Historical FP16 pilot artifacts, including node JSON, NPZ files, and run metadata. |

The four JSON files in `prompts/` also have byte-identical copies at the repository root for the existing Colab checks. The FP32 candidate loader and precision test also have byte-identical root copies for the project regression tests. Keep each pair in sync if an input or script is updated.

The [FP32 Stage 1 samples and audit](../shared_traces_fp32_stage1/) contain the later calibration and evaluation NPZ files and strict gate logs. See the [precision guide](../medusa-proj-skeleton/PRECISION.md) for the current collection and validation workflow.
