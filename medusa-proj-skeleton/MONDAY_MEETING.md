# Monday meeting update

## What Alex has completed

The laptop can now run the real Medusa model on its GPU. A separate project
Python environment contains the required CUDA and model libraries.

The code can choose a fixed tree or dynamically choose which guesses to check.
The selection code has been checked at all five required budgets. The decoder
matches official Medusa on all five notebook prompts, including the number of
tokens accepted in every round. It reproduces the notebook's 232 tokens across
93 rounds (2.49 tokens per round).

A cached ordinary-greedy baseline and benchmark now measure generation fairly.
The code handles equal token limits, warmups, exact output comparisons, and
separate tree-overhead profiling. Thirty-six automated tests pass.

The comparison pipeline generates tables, plots, and prompt-level confidence
intervals. All five methods and five budgets have been exercised with synthetic
test data. Those outputs validate the pipeline, not the research hypothesis.

Real local benchmarks of default Medusa and MEDUSA+ have run at all five dynamic
budgets. MEDUSA+ achieved about 0.82-0.93 times cached-greedy throughput on the
small five-prompt test. More accepted tokens did not produce a speedup here.
Two budget settings differed on the same story prompt. The first difference
was traced to an exact tie in the dynamic verification logits.

## What remains

1. Receive Lucia's real calibration and evaluation traces, with collection settings.
2. Receive Ada's fitted scorers and saved fixed trees in the agreed formats.
3. Run the full matched-budget comparisons on the real held-out traces.
4. Have Lucia run the validated benchmark on the designated T4 setup using the
   agreed MT-Bench prompts and output limits.
5. Write the results and conclusions from those experiments. The methods section
   already has a draft.
6. Review the code with the team on GitHub branch
   `codex/alex-tree-performance` before merging into `main`.

## Suggested spoken update

"I have the tree-selection and decoding code working on my laptop GPU. It
reproduces the official Medusa results on all five notebook prompts, including
the acceptance lengths. I also have a cached greedy baseline and a benchmark
that records speed, selection overhead, and output agreement.

I tested the comparison pipeline with synthetic data and ran a small real GPU
benchmark. It did not show a speedup on my laptop, and I documented two numerical
tie cases rather than dropping the mismatches. These are preliminary checks;
we still need the full real-data comparison and the final T4 timing runs.

My next step is to plug in Lucia's traces and Ada's trained models. Can we confirm
when the pilot files will be available and check that our formats and settings
match?"

## Items to settle in the meeting

- Delivery dates and locations for pilot traces, full traces, scorers, and trees.
- Exact model settings, dataset revisions, prompt splits, and output limits.
- Who will run the final T4 benchmark and when.
- Who will assemble the final report.

Bring `validation/alex/README.md` for the evidence and `README_ALEX.md` for the
run instructions. Present the work as implementation and preliminary validation
completed; the final experiments and research conclusions remain open.
