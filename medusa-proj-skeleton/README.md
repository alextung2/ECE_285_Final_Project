# Conditional Acceptance Calibration for Dynamic Medusa Decoding

ECE 285 project skeleton (Ada Qi, Alexander Tung, Zhuoting Yang).

Research question: with the Vicuna-7B v1.3 backbone and Medusa-1 heads frozen, do
parent-conditional acceptance estimates `q_v(x) = P(S_v | S_u, x)` select better
trees than marginal-product scores (Medusa / MEDUSA+) and a direct acceptance
classifier (C2T-inspired) at the same tree budget B?

## Layout

```
configs/default.yaml        all knobs (model, prompt sets, budgets, estimator hyper-params)
src/mcal/
  traces.py      TraceSet: the one data format (gold ranks + head top-K probs) + synthetic generator
  features.py    node batches, conditional / candidate-pool training sets, shared feature map
  estimators.py  MedusaAccuracy, MedusaPlusProb, CountConditional, LogisticConditional, DirectClassifier
  tree.py        budgeted best-first selection (ancestor-closed), fixed trees, medusa_choices export
  evaluate.py    trace-simulated accepted tokens, Brier/ECE, reliability, prompt-level bootstrap CIs
  diagnose.py    Step 1: product-of-accuracies vs observed joint acceptance
  model.py       load frozen Medusa checkpoint (fp16 or 4-bit)          [GPU]
  collect.py     greedy generations -> verification traces              [GPU]
  decode.py      online greedy Medusa loop with pluggable (dynamic) tree [GPU]
  data.py        AlpacaEval (calibration), MT-Bench (eval), HumanEval (optional)
scripts/
  00_gpu_check.py       Week 1   GPU memory check; load model; greedy vs Medusa sanity run
  01_collect_traces.py  Week 2   traces for calib (AlpacaEval) and eval (MT-Bench)
  02_diagnose.py        Week 2   product-score error and its direction
  03_fit_estimators.py  Week 3   fit count / logistic / direct; calibration table + reliability plot
  04_compare_trees.py   Wk 4-5   matched-budget comparisons (trace-simulated) with bootstrap CIs
  05_benchmark_speed.py Wk 6-7   online tokens/s, speedup vs greedy, tree-build overhead, agreement
tests/test_core.py      fast tests for the numpy core
```

## Quick start (no GPU needed)

```bash
pip install -r requirements.txt
pytest -q
bash run_synthetic.sh          # runs 01-04 on synthetic traces -> outputs/synthetic/
```

Synthetic traces make heads positively dependent across depth (a shared
"predictability" latent), so you can see the effect the project is about before
any real data exists. Set `traces.synthetic.dependence=0` and the conditional
estimator should collapse onto the marginal product (a good sanity check).
Synthetic numbers are for debugging only; never put them in the report.

## Real pipeline

```bash
pip install -r requirements-gpu.txt
python scripts/00_gpu_check.py --load                       # add --set model.load_in_4bit=true if short on memory
# put FastChat's MT-Bench question.jsonl at data/mt_bench_question.jsonl
python scripts/01_collect_traces.py --split calib
python scripts/01_collect_traces.py --split eval
python scripts/02_diagnose.py
python scripts/03_fit_estimators.py
python scripts/04_compare_trees.py
python scripts/05_benchmark_speed.py
```

## Key design decisions (worth a sentence each in the report)

* Greedy only. A node is accepted iff its prefix equals the backbone's greedy
  continuation, so a trace step is just `gold_rank[j]` (rank of the greedy token
  in head j's top-K) plus head probabilities. Accepted tokens for tree T is
  `1 + #accepted nodes in T`; a calibrated scorer predicts `1 + sum_v P_hat(S_v)`.
* Offline traces. Heads only see the hidden state of the last accepted token,
  so one teacher-forced pass over a greedy response gives the drafts Medusa
  would see at every position. Validate against online accepted tokens (05).
* Matched comparison. All methods share candidates (top-10 per head = Medusa's
  `TOPK`), `select_tree` and B. LogisticConditional and DirectClassifier share
  `node_features` and model class; only the target differs (S_v given an
  accepted parent vs. full-prefix S_v on a MEDUSA+ candidate pool).
* Monotone scores. Products of factors in [0, 1] can't exceed parent scores,
  so best-first gives an ancestor-closed top-B tree. The direct classifier is
  capped at its parent's score (`monotone: true`) for the same reason.
* Splits by prompt, never by step; bootstrap CIs resample prompts.
* Trace-only results (02-04) are not speedups. Only 05 measures wall-clock.

## Week-1 TODOs to verify on the GPU

1. `00_gpu_check.py --load`: model loads; Medusa output is identical to plain greedy.
2. `collect.py`: teacher-forced pass with `past_key_values=None` works on
   Medusa's KV-llama class, and `greedy_agree` printed by 01 is ~1.0.
3. Compare trace-simulated accepted tokens for the default Medusa tree with the
   online value from `00_gpu_check.py` (should be close).
4. `decode.DynamicTree` rebuilds Medusa buffers each step (cached by tree
   shape). Its Python cost is part of the "tree-building overhead" metric; if it
   dominates, vectorize `select_tree` or move it to torch.

## Extending

* New scorer: subclass `Scorer`, implement `fit` and `factor` (or `child_scores`
  + `path_scores`), add it to `REGISTRY`, then to the plans in 03-05.
* New feature: add a column in `features.node_features` (both learned models get it).
* Count table on full path instead of parent rank: change the index in `CountConditional`.
