# Tree selection and online performance methods

We compare ancestor-closed draft trees at budgets of 8, 16, 32, 63, and 96 nodes.
The backbone token is not counted in the draft budget. Candidate ranks span the
top ten tokens from each Medusa head. Best-first expansion chooses the highest
scoring available node and adds its children to the frontier. Every ancestor is
therefore selected before its descendants. Scorers must return finite nonnegative
path scores no greater than the parent's score. Impossible budgets are rejected.

Fixed policies use a predefined collection of rank paths, while dynamic policies
recompute paths from the current head predictions. Fixed-tree files use the
upstream Medusa choices format. Dynamic online evaluation uses the same selection
function as offline evaluation, with candidate ranks aligned to upstream logits.

For offline evaluation, the outcome at each recorded step is one backbone token
plus the length of the selected prefix matching the recorded gold ranks. We report
mean accepted tokens per verification step. Confidence intervals resample whole
prompts, preserving correlation among steps belonging to the same prompt. Paired
method differences use the same sampled prompt groups. Offline acceptance results
do not measure execution speed.

Online decoding uses the frozen quantized backbone and fp16 prediction heads.
The static implementation is checked against the official generator using exact
IDs and per-round acceptance lengths. All methods stop at the first EOS or the
same maximum new-token limit. The ordinary greedy reference reuses the upstream
KV cache and does not perform an unused forward pass after its final token.

Runtime measurement includes prefill and decoding and excludes loading,
tokenization, cache allocation, and fixed-buffer preparation. Dynamic selection
and buffer construction remain in the measured path. Warmups are excluded, and
method order is randomized within each prompt. Aggregate token throughput is
compared against cached greedy on the same prompts. Complete output agreement is
reported separately; a matching prefix with unequal length does not count as an
exact match. Quantization-related disagreements are recorded rather than treated
as a reason to discard the run.

Tree construction overhead is measured in a separate synchronized profiling run,
so additional synchronization does not alter the runs used for speedup estimates.
The final results section requires real held-out traces, fitted scorer handoffs,
and measurements on the designated T4 environment. Synthetic pipeline checks and
laptop measurements establish implementation readiness, not those final results.
