# Conditional acceptance calibration for Medusa decoding

[Final report](reports/final/ECE285_Final_Report.pdf) · [Study protocol and reproduction commands](reports/final/protocol/README.md) · [Evidence and report sources](reports/final/)

This ECE 285 study holds the frozen Medusa-Vicuna-7B checkpoint, five-head candidate space and tree budgets fixed while comparing five branch-scoring rules. The full replay covers 4,280 complete-horizon states from all 80 MT-Bench first turns. Conditional counts improved the fixed marginal tree by 0.019–0.044 accepted tokens per state across five budgets. The dynamic conditional model's paired differences from MEDUSA+ were small, with zero inside each per-budget 95% interval.

On the eight-question GPU panel, all five policies matched the cached-greedy output at budgets 16 and 63. Measured throughput ranged from 0.776 to 0.968 times cached greedy.

## Where to find things

| What you want to do | Start here |
| --- | --- |
| Reproduce the final study | Follow the [experiment commands](reports/final/protocol/README.md) for collection, calibration, evaluation and GPU timing. |
| Understand how the scorers were fitted | Open [Step 2 Calibration Integrated](Step%202%20Calibration%20Integrated.ipynb). The [final calibration runner](reports/final/run_calibration.py) configures this notebook for the real traces. |
| Inspect how trees are built and evaluated | Browse the [implementation](medusa-proj-skeleton/src/mcal/), [experiment scripts](medusa-proj-skeleton/scripts/) and [tests](medusa-proj-skeleton/tests/). |
| Trace a reported number back to its data | The [final evidence directory](reports/final/evidence/) contains replay comparisons, calibration tables and per-prompt GPU measurements. |
| Review the earlier work | See the [midterm report](reports/midterm/) and [Stage 1 pilot traces and logs](shared_traces_fp32_stage1/). |

## Results and data

- [All 25 matched-budget comparisons and prompt-bootstrap intervals](reports/final/evidence/tree_comparison.csv)
- [Path calibration on validation and evaluation prompts](reports/final/evidence/tables/calibration_path.csv)
- [Frozen prompt inputs, generated token IDs and replay traces](reports/final/evidence/cached_reference.zip)
- [Independent label reconstruction and coverage counts](reports/final/evidence/trace_audit.json)
- [Fitted scorers and fixed trees](reports/final/evidence/models_for_step3.zip)
- [GPU timing at 16 draft nodes](reports/final/evidence/speed_B16.csv) and [63 draft nodes](reports/final/evidence/speed_B63.csv)
- [Historical pilot and precision reproduction](reports/final/evidence/strict_precision_summary.json)

The archived midterm report and Stage 1 logs document the earlier teacher-forced protocol. The final experiment defines replay against one cached greedy trajectory and separately measures live agreement and throughput.

[Midterm report](reports/midterm/ECE285_Midterm_Report.pdf) · [Original Stage 1 traces](shared_traces_fp32_stage1/) · [Model, estimators and tree construction](medusa-proj-skeleton/src/mcal/)
