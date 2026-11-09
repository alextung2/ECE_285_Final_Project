# Medusa data collection updates

This folder groups the historical collection scripts, precision diagnostics, and environment notes uploaded by Lucia (`coconutcandy74`).

| Folder | Contents |
| --- | --- |
| [`collection/`](collection/) | Historical FP32 pilot and FP16 shared trace collection scripts. |
| [`diagnostics/`](diagnostics/) | Mismatch diagnostic script and the saved precision report for prompt 24. |
| [`environment/`](environment/) | Environment snapshot from the original pilot. |

## Trace results

- [FP32 Stage 1 samples and audit](../shared_traces_fp32_stage1/) contain the small calibration and evaluation NPZ files and the strict gate logs.
- [Historical FP32 pilot node traces](../pilot_node_traces_v2_fp32/) contain per-prompt JSON from the original pilot.
- [Historical FP16 shared traces](../shared_traces_fp16_v1/) contain the FP16 pilot artifacts.

## Root paths used by the project

- [Pilot prompts](../pilot_prompts.json), [training prompts](../train_prompts.json), [validation prompts](../validation_prompts.json), and the [dataset manifest](../data_manifest.json) stay at the paths used by the collection notebook.
- The [Medusa project notebook](../ece285_Medusa_Project.ipynb), [pilot collector](../collect_medusa_pilot.py), [shared loader](../load_model.py), [FP32 candidate loader](../load_model_candidate_4bit_weights_fp32_execution.py), and [precision test script](../test_medusa_precision.py) stay at their existing paths for validation and reproduction.

See the [precision guide](../medusa-proj-skeleton/PRECISION.md) for the current collection and validation workflow.
