param([switch]$SyntheticOnly)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
Set-Location $PSScriptRoot
$python = Join-Path (Split-Path $PSScriptRoot) '.venv\Scripts\python.exe'
& $python -m pytest tests -q
if ($LASTEXITCODE) { throw 'Tests failed' }
& $python scripts/01_collect_traces.py --synthetic --set traces.synthetic.n_prompts_calib=30 traces.synthetic.n_prompts_eval=10 traces.synthetic.steps_per_prompt=20
if ($LASTEXITCODE) { throw 'Synthetic traces failed' }
& $python scripts/03_fit_estimators.py --synthetic --set estimators.pool.n_steps=100 estimators.pool.budget=32
if ($LASTEXITCODE) { throw 'Synthetic scorer fixtures failed' }
& $python scripts/04_compare_trees.py --synthetic --set compare.eval_steps=200
if ($LASTEXITCODE) { throw 'Synthetic comparisons failed' }
if (-not $SyntheticOnly) {
    & $python scripts/06_validate_official.py
    if ($LASTEXITCODE) { throw 'Official equivalence failed; do not benchmark yet' }
    & $python scripts/05_benchmark_speed.py --set speed.prompt_file=data/smoke_prompts.json speed.n_prompts=5 speed.warmup_prompts=1 speed.max_new_tokens=64
    if ($LASTEXITCODE) { throw 'GPU smoke benchmark failed' }
}
