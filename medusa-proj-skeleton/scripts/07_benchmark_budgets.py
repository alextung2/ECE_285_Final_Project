"""Run the independent GPU policies across all five budgets, loading weights once.

Use the same CLI as 05_benchmark_speed.py. Defaults still expect MT-Bench;
for a small independent smoke run, pass speed.prompt_file=data/smoke_prompts.json.
"""
import importlib.util
import sys
from pathlib import Path
from _common import parse


def main():
    args, cfg = parse(__doc__)
    if args.synthetic:
        raise SystemExit('GPU timing cannot use synthetic mode')
    from mcal.model import load_medusa
    loaded = load_medusa(cfg)
    path = Path(__file__).with_name('05_benchmark_speed.py')
    spec = importlib.util.spec_from_file_location('benchmark_speed', path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    original = sys.argv[:]
    try:
        for budget in cfg['compare']['budgets']:
            sys.argv = [str(path), '--config', args.config, '--set', *args.set, f'speed.budget={budget}']
            print(f'Running budget {budget}', flush=True)
            benchmark.main(model_and_tokenizer=loaded)
    finally:
        sys.argv = original

if __name__ == '__main__':
    main()
