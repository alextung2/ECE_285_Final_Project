# Cached-reference experiment

The replay target is agreement with the next five draft tokens in a cached greedy continuation. The collector records the five head rankings and the backbone choice in the same model call. A stored path matches if every rank selects its corresponding future token. This definition allows offline comparison on any complete horizon, including prompts for which another execution shape changes a token. Online equivalence is a second empirical endpoint.

## Frozen choices

- Use the project NF4 checkpoint with FP32 compute after FP16 loading, TF32 disabled, ten candidates per head, and at most 64 new tokens. Keep greedy EOS termination. A response shorter than six tokens has zero complete five-head horizons. Retain its record and report zero-state prompts explicitly.
- Use the prepared prompt order and pinned dataset bytes. Exclude the training duplicate of MT-Bench 81 before collection. Attempt 199 training, 50 validation and 80 evaluation prompts. Fit on training IDs, tune on validation IDs and score all eligible evaluation states at budgets 8, 16, 32, 63 and 96.
- Run the integrated notebook grids and use the common 64-node MEDUSA+ candidate pool for Brier and reliability. The repository tree comparison uses 2,000 paired, prompt-level bootstrap replicates. Treat the position-weighted mean accepted count, including one backbone token, as the primary offline endpoint.
- For live comparisons, select the first MT-Bench question in each of its eight categories. Use question 82 as the separate warmup. Use budgets 16 and 63. Keep every discrepancy. Only interpret throughput as equivalent-output speed on a panel where every measured output for that policy and budget agrees with greedy.
- For the new panel, require exact cached-reference and final output IDs for an output-equivalence claim. For offline label validation require identical visited prefixes and selected-node accept/reject decisions at every complete-horizon round. Record selected and unselected rank changes, as both can affect the interpretation of replay. The original strict gate still runs separately and reports all-rank equality.

## Reproduction

From the repository root, use the project GPU environment and the pinned upstream Medusa checkout. Install pandas for the calibration notebook. The preparation script writes host-specific paths, so generate the experiment directory on the machine that runs it.

From `medusa-proj-skeleton`:

```text
python scripts/11_prepare_experiment.py --out outputs/final_frozen --max-new-tokens 64
python scripts/12_collect_cached_reference.py --config outputs/final_frozen/config.yaml
```

From the repository root:

```text
python reports/final/run_calibration.py --traces medusa-proj-skeleton/outputs/final_frozen/cached_reference --out medusa-proj-skeleton/outputs/final_tuned
```

From `medusa-proj-skeleton`:

```text
python scripts/02_diagnose.py --config outputs/final_frozen/config.yaml --set out_dir=outputs/final_tuned/real
python scripts/04_compare_trees.py --config outputs/final_frozen/config.yaml --set out_dir=outputs/final_tuned/real estimators.manifest=outputs/final_tuned/real/split_manifest.json compare.eval_steps=null
python scripts/13_verify_cached_panel.py --config outputs/final_frozen/config.yaml --trace-dir outputs/final_frozen/cached_reference --panel ../reports/final/protocol/timing_ids.json --set out_dir=outputs/final_tuned/real "compare.budgets=[16,63]"
python scripts/07_benchmark_budgets.py --config outputs/final_frozen/config.yaml --set out_dir=outputs/final_tuned/real "compare.budgets=[16,63]" speed.n_prompts=8 speed.warmup_prompts=1 speed.prompt_file=../reports/final/protocol/timing_stratified.json
```

Use the historical teacher-forcing protocol for the precision reproduction. Run `scripts/09_validate_precision.py` with the frozen config and a fresh output directory, first with its built-in seven cases and then with the five archived failed pilot inputs in `pilot_failures.json`. A failed gate returns exit code 1 and saves its observations.

For the small historical pilot, use the Stage 1 calibration and evaluation NPZ files, the default config with `model.precision_mode=nf4_fp32_from_fp16`, and the four/one split in `../reports/final/pilot/manifest.json`. Run scripts 02, 03 and 04. Sample models depend on the Colab runtime contract and are unsuitable for this laptop's live experiment.
