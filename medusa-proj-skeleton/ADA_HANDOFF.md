# Ada integration and the GPU handoff

## Current data status before merge (2026-11-02)

Main at 66d6232 now includes all 20 FP32 pilot node JSON files (32,806
nodes, 20 recorded exact-output agreements), the prompt-24 precision report,
and the original FP16 shared pilot NPZ. Earlier missing-upload notes below
describe the state when those checks were performed.

The FP16 NPZ has 498 complete rounds from 20 prompts; all six arrays reproduce
exactly from the recorded JSON. Default-tree acceptance replay agrees on all
547 recorded rounds, but prompt 24 differs from uncached greedy at token 32.
The FP32 node JSON still lacks complete TraceSet arrays. Full compatible FP32
training/validation/evaluation NPZ collection and representative dynamic-policy
GPU validation remain pending. Merging this CPU-tested code does not certify
GPU correctness or complete the research experiments.


Ada's main-branch upload (`d5ca58c`) contains five scorers, tuning/ablation code,
and a saved bundle. Its settings explicitly identify synthetic training data.
The original notebook, ZIP and root-level module snapshots remain unchanged.

## Active code

Use `Step 2 Calibration Integrated.ipynb` from the repository root in a fresh
kernel. Install `medusa-proj-skeleton/requirements.txt` for CPU dependencies.
The integrated notebook imports `medusa-proj-skeleton/src/mcal` rather than
writing private copies of the package. Repository-root imports route there too.
Do not run the historical notebook in the working checkout: its writefile cells
overwrite module files. Run it in a separate copy if reproducing old evidence.

The canonical package now includes Ada's grouped features and ablations,
precomputed conditional training batches, and union candidate pools for direct
classifier training. It retains the stricter budget/tree and precision checks.
Ada also corrected synthetic top-K ordering; newly generated synthetic fixtures
use that correction, while previous fixtures remain historical evidence.

Two feature columns changed order: relative confidence and log path probability.
The parent-rank cap also now keeps high ranks separate from the root bucket.
Fitted logistic/direct models carry `feature_schema=ada_grouped_v1`; loading or
scoring a model without that schema fails. Do not add a schema tag to old pickles:
refit using the integrated notebook or CLI. Static models and fixed trees still
require matching execution contracts at pipeline handoff. This prevents Ada's
untagged synthetic ZIP from being mistaken for a final real-data model bundle.

## CPU workflow

The integrated notebook preserves Ada's hyperparameter search, direct-pool
selection on validation prompts, and feature ablations. It exports tagged
scorers/trees, exact input traces, a split manifest, settings and a portable code
bundle to a fresh `OUT_DIR/synthetic` or `OUT_DIR/real`. No historical files are
overwritten. Path calibration is refreshed after choosing the final direct model.
Synthetic tuning values are not final research settings.

For CLI fitting, `estimators.pool.self_retrain=true` enables one round on the
deduplicated union of reference and direct trees at the configured pool budget.
This is configurable training, not a substitute for the notebook's validation
search over multiple pool variants. CLI training records its configuration,
split IDs, manifest hash, feature schema and execution contract.

From `medusa-proj-skeleton`, a small CPU smoke run is:

```sh
python scripts/01_collect_traces.py --synthetic --set out_dir=outputs/ada_smoke traces.synthetic.n_prompts_calib=20 traces.synthetic.n_prompts_eval=8 traces.synthetic.steps_per_prompt=20
python scripts/03_fit_estimators.py --synthetic --set out_dir=outputs/ada_smoke estimators.pool.n_steps=40 estimators.pool.self_retrain=true
python scripts/04_compare_trees.py --synthetic --set out_dir=outputs/ada_smoke compare.eval_steps=20
```

To compare integrated notebook exports, set `out_dir` to its base output folder
with `--synthetic`, or to its `real` subfolder without that flag. For real data,
also set `estimators.manifest` to its `split_manifest.json` and set the matching
`model.precision_mode`. The saved traces have the default `traces/calib.npz` and
`traces/eval.npz` paths. Evaluation traces are required by the comparison CLI.

Real fitting and comparison audit inputs and require a manifest containing
`train_prompt_ids` and `validation_prompt_ids`. Lists must be nonempty, unique,
disjoint and exactly cover collected calibration IDs. There is no random
fallback for incomplete real pilot exports. Keep the final evaluation prompts
separate from all tuning decisions. Unknown cross-dataset text overlap still
requires original prompt manifests; numeric IDs alone cannot establish this.

## What remains for the GPU laptop

CPU integration does not certify numerical equivalence on CUDA. The precision
loader and decoding algorithms were not changed by this integration. On the GPU
laptop, use the same PR checkout and dependencies, run the correctness gate in
PRECISION.md, collect/audit compatible real traces, and refit the models. Then
validate trained-policy output agreement and run latency benchmarks separately.
The provisional FP32 recommendation remains conditional on those checks.

Lucia still needs to provide complete TraceSet arrays plus settings and prompt
manifests. Her existing node JSON is insufficient for lossless conversion.
Ada should use the integrated notebook for the real-data rerun; she does not need
to repeat her historical synthetic experiments. No synthetic result here is a
research result or GPU speedup.

## Executed checks

- Full CPU suite: 69 passed, 2 skipped (unavailable Medusa/FastChat).
- Feature arrays match Ada's committed source exactly for both rank caps and
  every single-group ablation. All five scorers survive save/reload and generate
  valid trees at budgets 8, 16, 32, 63 and 96.
- All integrated notebook code cells ran on CPU with 20 calibration prompts x
  20 steps, 8 evaluation prompts x 20 steps, pool_steps=40 and quick_eval_steps=20.
  Tuning, direct variants, ablations and export completed. Notebook output cells
  are intentionally cleared in Git; original notebook outputs are preserved.
- Both CLI-trained and notebook-exported models completed all 25 scorer/budget
  comparisons on 20 synthetic evaluation steps. Compile and whitespace checks
  passed. No GPU validation or real-trace experiment was run.
