# Alex's tree selection and performance work

This checkout contains Alex's implementation and validation tools. Run commands
from `medusa-proj-skeleton` using `../.venv/Scripts/python.exe` on this laptop.
The local branch is `codex/alex-tree-performance`.

## What can run without partner files

- Tree-selection and decoder regression tests.
- A complete synthetic comparison pipeline with all five methods and budgets.
- Official Medusa token and per-round acceptance validation.
- Cached greedy, default Medusa, and dynamic MEDUSA+ GPU benchmarks.

Synthetic fitted scorers are test fixtures only. They are isolated under
`outputs/synthetic`, and must not be used as the trained project models.

## Environment

Python 3.11, PyTorch 2.2.2 with CUDA 11.8, transformers 4.36.2,
bitsandbytes 0.43.3. The backbone uses NF4 with double quantization; lm_head
and all Medusa heads remain fp16. `environment_alex_windows.txt` records the
installed packages. The upstream Medusa checkout is pinned by the installation
instructions below; its code is not modified by this project.

The laptop has an RTX 3080 Ti Laptop GPU with 16 GiB memory. Results from this
hardware are local preliminary measurements, not the final Colab T4 speed table.

To reproduce in a fresh environment, from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python -m pip install -r medusa-proj-skeleton/requirements-alex.txt
.\.venv\Scripts\python -m pip install --no-deps git+https://github.com/FasterDecoding/Medusa.git@e2a5d20c048a9b0a4092e6933c34313687422518
```

For faster initial weight downloads, optionally install `hf_transfer` and set
`$env:HF_HUB_ENABLE_HF_TRANSFER='1'`. Set `$env:PYTHONUTF8='1'` for console output.
Model weights are downloaded into Hugging Face's user cache, not committed.

## Run the checks

```powershell
cd medusa-proj-skeleton
.\run_alex_checks.ps1 -SyntheticOnly
.\run_alex_checks.ps1
```

The full script runs tests, synthetic comparisons, official equivalence, and a
small GPU benchmark. It stops if a required check fails.

Individual GPU commands:

```powershell
..\.venv\Scripts\python scripts/06_validate_official.py
..\.venv\Scripts\python scripts/05_benchmark_speed.py --set speed.prompt_file=data/smoke_prompts.json speed.n_prompts=5 speed.warmup_prompts=1 speed.max_new_tokens=64
```

The validation script runs the actual upstream generator on five prompts,
capturing IDs and accepted lengths via a temporary hook. It uses the notebook's
five prompts and 64-token limit. Our output is truncated at first EOS;
raw acceptance lengths are retained for the upstream comparison. Trained head
weights are checked against the saved checkpoint. Cached and uncached greedy
outputs are compared separately; numerical disagreements are recorded.

The benchmark defaults to `medusa_default` and `medusa_plus`. It needs no saved
scorers for either. It checks full sequence equality with cached greedy, including
length, and records mismatches without stopping. Quantization-related differences
are a measured result. Default Medusa's native tree size is reported explicitly;
it is not relabeled as an arbitrary requested budget.

## Timing definitions

Main timing covers prefill and generation, with synchronization at its boundaries.
It excludes loading, tokenization, KV-cache allocation, and fixed-tree buffer setup.
Dynamic tree policy and buffer construction remain inside timed generation.
Methods are randomly ordered within each prompt to reduce ordering bias. Warmup
prompts are excluded. Throughput is total generated tokens divided by total time;
speedup divides this throughput by the cached greedy throughput on the same prompts.

Tree overhead is profiled separately with synchronization around policy and buffer
construction. It is reported for the first measured prompt; these instrumented
wall times never enter the speedup calculation. This separation avoids treating
asynchronous CUDA launch time as completed work or contaminating main timing with
extra per-round synchronizations.

Outputs under `outputs/results` include CSV summaries, per-prompt IDs and timings,
configuration, package versions, GPU identity, and a prompt-set hash. A short local
smoke run validates functionality; it does not establish final research conclusions.

## Partner handoffs

Place Lucia's `calib.npz` and `eval.npz` in `outputs/traces`. Place Ada's
`medusa_acc.pkl`, `count_cond.pkl`, `logistic_cond.pkl`, `direct.pkl`, and
`medusa_plus.pkl` in `outputs/models`, along with fixed-tree JSON files for
`medusa_acc` and `count_cond` at 8, 16, 32, 63, and 96 nodes.

```powershell
..\.venv\Scripts\python scripts/04_compare_trees.py --set compare.eval_steps=null
..\.venv\Scripts\python scripts/05_benchmark_speed.py --set 'speed.methods=[medusa_default,medusa_acc,count_cond,medusa_plus,direct,logistic_cond]'
```

For the full benchmark, provide the MT-Bench first-turn question file at
`data/mt_bench_question.jsonl`. Pin its revision and output length with Lucia.
Run the benchmark at each required budget on the final T4 environment. Supply
exact model/data revisions and verify collection settings match before claiming
comparability with Lucia's traces.

The existing config key `estimators.val_frac=0.8` is historically misnamed: it is
passed as the *training fraction*. Avoid changing Ada's split without coordination.
The trace-collection defaults are not a substitute for Lucia's pinned collection
procedure. The original Medusa accuracy-product comparison is a rebuilt matched-
budget baseline; `medusa_default` is the separate official native tree.

To run all five budgets with one model load:

```powershell
..\.venv\Scripts\python scripts/07_benchmark_budgets.py --set out_dir=outputs/laptop_smoke speed.prompt_file=data/smoke_prompts.json speed.n_prompts=5 speed.warmup_prompts=1 speed.max_new_tokens=64
```

Completed-session results and remaining dependencies: [validation report](validation/alex/README.md).
