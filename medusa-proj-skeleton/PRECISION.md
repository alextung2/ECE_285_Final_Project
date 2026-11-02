# Precision investigation and handoff

Investigated from main `0bba339b8359e69faa736df6d1d5cc0b27c9e87a` (merged
tree selection/benchmarking). Original notebook outputs and `validation/alex/`
remain historical evidence, not results of this change.

## Decision

**Recommend NF4 weight storage with FP32 execution for the final experiments,
conditional on the correctness gate below.** This is the strongest candidate
in the available evidence, not a claim that the final configuration is validated.
Do not collect final labels or report final comparisons until the exact loader
is agreed and the broader checks pass. Keep FP16 as the default historical
reproduction mode and an explicitly separate comparison arm. Do not mix its
traces, fitted scorers, or timing baseline with FP32 artifacts.

The new `nf4_fp32_from_fp16` mode is a precise, independently defined candidate.
It is **not asserted to reproduce Lucia's missing loader**. If her source reveals
different promotion, quantization, TF32, or buffer behavior, reconcile that
difference and rerun the gate before choosing the final mode. The safe outcome
of a failed gate is to retain the failure and investigate/recollect, not to
weaken the gate or discard prompts.

## What the evidence establishes

All 18 notebook cells, their stored text outputs, the two root Python scripts,
the collector/decoder/model code, and Alex's validation reports were inspected.

| Evidence | Finding | Limit |
|---|---|---|
| Notebook cells 8-11 | NF4, double quantization, FP16 compute and nonquantized FP16 heads; five prompts agree; 232 tokens / 93 rounds | Five fixed-tree prompts do not cover dynamic trees |
| Cells 13-15 | Pilot prompt 24, "What breed dog is smallest?", diverges at token 32; official Medusa reproduces the collector | Establishes this is not only a custom-decoder bug |
| Cell 15 | Same-prefix cached and uncached logits favor `2` over `1` by 0.015625; official tree logits favor `1` by 0.015625 | Numerical ordering reversal, not an exact tie in this case |
| Cell 16 | `fp32_lm_head` fails; `4bit_weights_fp32_execution` passes prompt 24 | Candidate source is absent from GitHub |
| Cell 17 | Broader candidate pilot: 20/20 exact agreement, 32,806 node rows, 8,950 conditional rows, 680 accepted-prefix rows | Stored notebook evidence, not independently rerun; fixed tree and 64-token horizon only |
| `validation/alex/official_validation.json` | Five notebook prompts match official tokens and raw/effective acceptance lengths; cached/uncached/static agreement | Historical FP16 run |
| `dynamic_divergence_B8.json`, `dynamic_divergence_B96.json` | Story prompt at index 50: cached period/anywhere logits 23.03125/23.0; dynamic 23.015625/23.015625 | These two known failures must be in the new gate |

The notebook uploads `test_medusa_precision.py` and later executes
`load_model_candidate_4bit_weights_fp32_execution.py` from Drive. Neither source
is embedded in the notebook or present in this checkout. Searches of connected
Drive for that loader and `ECE285_Medusa` returned no results. Consequently the
exact tensors, layer compute flags, cast timing, and TF32 settings changed by
Lucia's candidate **cannot be determined from available evidence**. Its name and
successful output are not an implementation specification. Her printed 5.03-GiB
"model loaded" line occurs before the candidate-mode line and cannot be used as
the post-promotion footprint.

## Cause and dtype distinctions

Greedy verification is lossless in exact arithmetic when the same conditional
model is evaluated. Cached token-wise, full-prefix, teacher-forced, and tree
forwards use different matrix shapes and reduction orders. Finite-precision
rounding can change a near-tied argmax, after which the generated context diverges.
FP32 reduces this risk; it does not guarantee identical decisions across shapes,
backends, hardware, or every possible prompt. Exact token agreement alone also
does not prove every rejected node's acceptance label is correct.

NF4 describes packed backbone weight storage and its quantization error. It does
not specify the floating arithmetic used after dequantization. In
[bitsandbytes 0.43.3 Linear4bit](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.43.3/bitsandbytes/nn/modules.py),
`compute_dtype` controls arithmetic inputs and outputs are cast back to the input
activation dtype; the first input can also set the computation dtype. Setting
only this flag while retaining FP16 activations is not full FP32 execution.
`lm_head` and `medusa_head` are excluded from quantization and have their own
floating weights. Casting already-rounded logits to FP32 for softmax cannot
undo the forward-pass rounding. The
[pinned Medusa loader](https://github.com/FasterDecoding/Medusa/blob/e2a5d20c048a9b0a4092e6933c34313687422518/medusa/model/medusa_model.py)
loads backbone and Medusa head checkpoints separately.

| Dimension | NF4 + FP16 | NF4 + FP32 candidate |
|---|---|---|
| Greedy / verification agreement | Known failures despite five-prompt success | Lucia's 20-prompt fixed-tree success is encouraging; new implementation and dynamic budgets unverified |
| Stability | Near ties can flip or collapse to ties | More arithmetic precision; quantization error and shape-dependent rounding remain |
| Memory | Historical model tensors ~5.03 GiB | Floating heads, activations and KV cache grow; packed NF4 weights stay packed |
| Recorded diagnostic peak | LM-head-only trial 6.49 GiB | Lucia's broader trial 9.34 GiB; different diagnostic states, not controlled capacity measurements |
| Performance | Historical laptop dynamic throughput 0.82-0.93x cached greedy | No measured FP32 latency evidence; expect a possible cost, particularly on T4; measure separately |

Reliable acceptance labels take priority over preserving an apparent speedup.
Every policy and the cached-greedy baseline must use the same mode, checkpoints,
software, prompt set, token limits and hardware within each comparison. Report
memory and throughput separately per mode; never normalize FP32 Medusa against
an FP16 baseline. Diagnostics and observers are excluded from latency runs.

## Implemented modes and guards

Both modes load and quantize the same FP16 source into NF4 double-quantized
weights. `dtype: float16` is the **initial loading/quantization** dtype, not the
final execution dtype. Conflicting values are rejected.

* `nf4_fp16`: retain FP16 floating parameters and compute, nonquantized FP16 heads.
* `nf4_fp32_from_fp16`: after loading, promote every nonpacked floating parameter
  and registered floating buffer to FP32 (including embeddings, norms, rotary
  buffers, LM head and Medusa heads); set each Linear4bit computation dtype to
  FP32. Preserve packed parameters and their quantization state. This retains
  initial FP16 rounding; it is not a fresh full-FP32 checkpoint load.

TF32 is explicitly disabled in both arms. There is no autocast wrapper.
The loader audits actual compute/head/parameter dtypes and packed NF4 state,
then records the spec, dependency versions, backbone revision, hashes of relevant
upstream files and actual loaded head tensors. Runtime audit details accompany
pilot settings and benchmark metadata. This detects known configuration and
artifact mismatches; it is not proof of cross-device numerical identity or a
complete weight-integrity attestation for custom local backbones.

Root `load_model.py` delegates to `mcal.model.load_medusa`. It now requires the
complete checkout, not a standalone copied loader. Select the mode through
`MEDUSA_PRECISION_MODE` for the pilot and `model.precision_mode` for the numbered
pipeline. Unknown legacy traces remain readable for historical inspection, but
cannot be concatenated with new real traces, fitted, compared or deployed through
the scripts without a matching execution contract. Synthetic and real artifacts
cannot be mixed. Fitted pickle objects and fixed-tree JSON carry the contract;
legacy fixed-tree lists are rejected by the comparison and benchmark scripts.

Offline features now rank **raw logits first**, exactly as upstream candidate
generation does, then gather FP32 full-vocabulary softmax probabilities. Ranking
softmax values can lose distinctions through underflow or rounding. This is
independent of choosing FP16 versus FP32.

Collection requires exact cached/static output agreement, teacher-forced argmax
agreement over the whole generated response (including the label tail), and
matching candidate ranks and node labels at visited static verification rounds.
It fails rather than accepting a 99.5% threshold. The dynamic GPU gate additionally
tests all five budgets. Matching labels at visited rounds is finite evidence,
not a theorem about unvisited contexts. Probability differences are recorded by
the observer; label/rank checks are exact, and no missing labels are synthesized.

## JSON pilot interoperability

**The existing pilot JSON is not losslessly convertible into the current
TraceSet.** Changing the file extension cannot fix the semantic gap.

| TraceSet field | Pilot availability |
|---|---|
| prompt_id | Present; retain original dataset IDs |
| position | Derivable for visited rounds from preceding `used_tokens`; round ID is not token position |
| topk_prob `[N,H,K]` | Only visited node confidences; unvisited head/rank entries missing |
| root_prob `[N]` | Not stored; a depth-1 parent confidence of 1.0 is a convention, not root probability |
| gold_rank `[N,H]` | Cannot infer all ranks from a finite fixed tree, rejected prefixes, or truncated continuation |
| topk_token `[N,H,K]` | Only visited candidates; full lists absent |
| execution provenance | Historical settings have loader hash but no complete audited dtype contract; source required |

Missing data is unknown, not zero probability, not a negative label, and not rank
`-1`. A failed prefix censors conditional labels downstream. Also, numerical
tree-dependent labels need not admit one global gold-rank path. The existing JSON
can support analysis/training restricted to its observed nodes with a dedicated
node-data estimator, but cannot fairly evaluate arbitrary new trees or directly
feed the current estimators. No such partial conversion is advertised here.

Preferred handoff: Lucia recollects `.npz` with `01_collect_traces.py` using the
agreed shared loader and the original split manifests. This is **new collection**,
not conversion of the 32,806 records. Alternatively, provide a richer export for
each eligible state with all H x K raw-logit-ranked token IDs and full-vocabulary
probabilities, root probability, H future cached-greedy token IDs/ranks (with
explicit outside-top-K status), exact prefix and generated position, prompt ID,
EOS/length eligibility, original tree paths and verification labels. Include
loader/test source, hashes, checkpoint/software revisions, dtype audit,
tokenizer/template, dataset revision and train/validation IDs. Validate its
gold-prefix labels against actual tree verification before writing NPZ; if they
disagree, a richer node-specific format or recollection is necessary.

Ada must refit on the selected-mode traces and use newly tagged scorer/tree
artifacts. FP16-trained or provenance-free scorers must not be silently reused.
Alex must rerun correctness before benchmarking. No one should edit original
pilot traces to add guessed settings or labels.

## Commands and release gate

Run from `medusa-proj-skeleton/` in the pinned Medusa GPU environment. Each mode
gets a fresh process and new output directory. FP32 requires no change to
`model.dtype`:

```sh
python scripts/06_validate_official.py --set model.precision_mode=nf4_fp32_from_fp16 out_dir=outputs/fp32_official
python scripts/09_validate_precision.py --set model.precision_mode=nf4_fp32_from_fp16 speed.max_new_tokens=64 out_dir=outputs/fp32_known_cases
python scripts/09_validate_precision.py --prompt-file /path/pilot_prompts.json --set model.precision_mode=nf4_fp32_from_fp16 speed.max_new_tokens=64 out_dir=outputs/fp32_pilot
```

Repeat with `nf4_fp16` into different directories; keep failures and token IDs.
Run at the intended final output/context lengths as well, with held-out
AlpacaEval, MT-Bench first turns, code, translation, and longer prompts. The
default gate has all five notebook prompts plus Chihuahua and lost-key cases;
it does not claim to replace the representative held-out set. Re-run on T4 and
the actual benchmark backend. Require zero token/rank/label mismatches, and
inspect probability-error reports before fitting. If complete H-token horizons
cannot be labelled, mark those cases explicitly rather than treating a skipped
case as a pass.

For reproducible split handoff, `traces.calib.prompt_file` and
`traces.eval.prompt_file` accept Lucia's JSON records with `prompt_id` and
`instruction`; this preserves her IDs and order rather than using the existing
dataset loader's first-N selection. Supply disjoint files and keep the original
dataset manifest alongside them. Collection records a content hash.

```sh
python scripts/01_collect_traces.py --split calib --set model.precision_mode=nf4_fp32_from_fp16 traces.calib.prompt_file=/path/train_prompts.json out_dir=outputs/fp32_final
python scripts/03_fit_estimators.py --set model.precision_mode=nf4_fp32_from_fp16 out_dir=outputs/fp32_final
python scripts/04_compare_trees.py --set model.precision_mode=nf4_fp32_from_fp16 out_dir=outputs/fp32_final
```

Collect the chosen evaluation split before fitting (the fit script reads both).
Use MT-Bench for the final evaluation; Lucia's Alpaca validation split is not
automatically the project's final evaluation set. The scripts reject old/mixed
contracts. For the node pilot, from the checkout root in Colab/Linux:

```sh
MEDUSA_PRECISION_MODE=nf4_fp32_from_fp16 python collect_medusa_pilot.py --project-dir /path/original_run --loader /path/checkout/load_model.py --output-dir /path/new_audited_pilot --num-prompts 20
```

Do not rerun historical notebook cell 17 against the revised collector: it uses
literal source replacement for the old standalone loader. Keep that notebook
as evidence; use the explicit command above for new runs.

Only after correctness passes, use `07_benchmark_budgets.py` with the same mode,
fresh output directory, agreed prompts and token limits. It uses cached greedy
in the same model, warmups, random policy order, exact output checks, and a
separate overhead profile. `09_validate_precision.py` uses observers and
teacher-forced passes and is **never a latency benchmark**. Diagnostic memory
peaks likewise must not be compared as isolated decode peaks.

## Verification in this change

CPU environment: Windows, Python 3.13, PyTorch 2.13.0+cpu. CUDA is unavailable;
`nvidia-smi` is absent. The real validation command exits with
`CUDA unavailable: real precision validation NOT RUN`. No GPU correctness,
FP32 memory/latency, actual pilot conversion, or final research experiment was
executed here. See `validation/precision_checks.md` for executed regression and
synthetic integration checks. The missing Drive loader/settings and representative
GPU gate remain required before certifying the final experimental configuration.
