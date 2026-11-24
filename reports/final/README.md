# ECE 285 final study

[Read the report](ECE285_Final_Report.pdf) · [Reproduce the experiments](protocol/README.md)

## Inspect the results

- [Matched-budget acceptance and paired prompt-bootstrap intervals](evidence/tree_comparison.csv) cover five methods on all 80 MT-Bench first turns.
- [Calibration and tuning tables](evidence/tables/) include path Brier score, ECE, direct-classifier pool comparisons, and the validation feature ablation.
- [Live panel observations](evidence/cached_verification.json) preserve round-level prefixes, changed candidate ranks and selected-node label discrepancies. [Historical failure reproductions](evidence/strict_precision_summary.json) summarize a separate teacher-forced check.
- The [real-data model bundle](evidence/models_for_step3.zip) contains the five fitted or parameter-free scorers, fixed trees, settings and implementation used for the full study. The smaller [pilot analysis](pilot/) used the archived Colab traces and a separate four/one calibration split.

## Verify the data without a GPU

The [portable input and trace archive](evidence/cached_reference.zip) includes the prepared prompt files, token IDs for all 329 attempted generations, individual prompt traces, consolidated arrays, and the eligible split manifest. From the repository root, extract into a fresh directory and run:

```text
python -m zipfile -e reports/final/evidence/cached_reference.zip replay_check
python reports/final/audit_replay.py replay_check --out replay_check/audit.json
```

Compare the result with the [committed audit](evidence/trace_audit.json). Both consolidated archive hashes also match the inputs recorded by the [tree comparison](evidence/tree_comparison_metadata.json).

The timed runs can be audited with `python reports/final/audit_timing.py reports/final/evidence`. The script recomputes throughput, accepted tokens, speed ratios and exact output agreement from the per-prompt records.

## Build the report

The editable source starts at [ECE285_Final_Report.tex](ECE285_Final_Report.tex). The PDF uses the [official NeurIPS 2026 style](https://neurips.cc/Conferences/2026/CallForPapers) with a course-report footer. It has nine pages of main text followed by references. Compile twice with pdfLaTeX from this directory. Run `python render_example.py` here to redraw the two-slot illustration. The reliability plot and measurement tables use the archived evidence. To regenerate those files after reproducing the study, run the following from the repository root:

```text
python reports/final/render_tables.py medusa-proj-skeleton/outputs/final_tuned/real --out reports/final/tables
```

The two original detailed November 9 Colab JSON logs were unavailable. The project retains the original aggregate logs and the new local retest evidence separately.
