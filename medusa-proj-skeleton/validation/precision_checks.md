# Precision change: executed checks

Date: 2026-11-02. Updated against main `54cfc8c00deeb7e42e5357d6c818ef4bb610e929`
(Lucia's FP32 source upload); initial base was `0bba339b8359e69faa736df6d1d5cc0b27c9e87a`.
This is a new CPU verification report. Nothing under `validation/alex/` or in
the original notebook was changed.

## Regression tests

`python -m pytest -q -rs`: **50 passed, 2 skipped**.

Of the 36 pre-existing cases, 34 passed and two require unavailable dependencies:
`test_choices_load_in_real_medusa_buffers` (Medusa) and the FastChat template
comparison. The 16 new cases passed. They cover explicit mode specifications,
conflicting/unknown modes, metadata round trips, mismatched/unknown/synthetic
contracts, rejecting node JSON as NPZ input, packed-weight-preserving promotion,
head dtype auditing, raw-logit ranking under softmax underflow, observer label
disagreement, cached/static mismatch, and teacher-forced mismatch in the label tail.
Promotion and verification tests use small controlled CPU tensors, not real
bitsandbytes GPU execution or downloaded Vicuna weights.

The alignment update adds two parametrized comparisons against the **actual
committed source**: the constants in `test_medusa_precision.py` and the promotion
section in `load_model_candidate_4bit_weights_fp32_execution.py`. Both agree with
our shared implementation for floating tensor values/dtypes, integer buffers,
packed-weight pointer/state preservation, Linear4bit flags and FP16-input LM-head
outputs. Repeated promotion installs only one LM-head input hook. A third new
case rejects pre-alignment contracts missing the hook field. The runtime audit
also rejects a missing hook. These CPU fixtures stand in for bitsandbytes layer
and parameter types; they do not execute real 4-bit CUDA kernels.

Environment: Windows, Python 3.13, torch 2.13.0+cpu, numpy 2.2.2,
scikit-learn 1.6.1, scipy 1.15.1, pytest 9.2.0. This differs from the pinned
Python 3.11 / torch 2.2.2 GPU environment; no claim of GPU compatibility testing.

## Synthetic pipeline integration

Executed sequentially for the initial PR version from `medusa-proj-skeleton/`
(not rerun for the later LM-head-hook change, which does not affect synthetic code):

```sh
python scripts/01_collect_traces.py --synthetic --set out_dir=../../synthetic_validation traces.synthetic.n_prompts_calib=20 traces.synthetic.n_prompts_eval=8 traces.synthetic.steps_per_prompt=20
python scripts/03_fit_estimators.py --synthetic --set out_dir=../../synthetic_validation estimators.pool.n_steps=100
python scripts/04_compare_trees.py --synthetic --set out_dir=../../synthetic_validation compare.eval_steps=40
```

All exited successfully. Generated 400 calibration and 160 evaluation steps;
fit/held-out calibration split was 320/80 steps. All five scorers saved and
loaded successfully with matching contracts. Both fixed-tree artifact types
were consumed in the new tagged format. All 25 method-budget combinations
completed (budgets 8, 16, 32, 63, 96), producing synthetic CSV/plot outputs.
These numbers verify plumbing only, not real trace compatibility, model quality,
or measured decoding speedups. Generated scratch artifacts are not committed.

## Static checks and historical preservation

`compileall` on source, scripts, tests and the root scripts completed. Both newly uploaded Lucia scripts
were also compiled successfully and remain byte-for-byte unchanged from main.
`git diff --check` passed. Git confirmed no changes to
`ece285_Medusa_Project.ipynb` or `validation/alex/`.

## Real validation: not run

`torch.cuda.is_available()` returned false; `nvidia-smi` was unavailable.

```sh
python scripts/09_validate_precision.py --set model.precision_mode=nf4_fp32_from_fp16 out_dir=outputs/lucia_alignment_unrun
```

Exited nonzero with `CUDA unavailable: real precision validation NOT RUN`.
No successful GPU validation report was produced. Real FP32 loader operation,
representative prompt agreement, offline/online labels, memory and throughput
remain unverified. The candidate loader source is now available and reconciled.
The actual pilot JSON/settings were not included in the upload, so no real pilot
was imported or converted. See [PRECISION.md](../PRECISION.md)
for the evidence, conditional recommendation and exact team handoff requirements.
