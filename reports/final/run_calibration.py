"""Run the existing calibration notebook against the cached-reference study.

Run from the repository root with Python and the CPU analysis dependencies.
The notebook source and its synthetic defaults are preserved.
"""
import argparse
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--traces", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
settings = dict(USE_SYNTHETIC=False, DATA_DIR=args.traces.resolve(),
                MANIFEST_FILE=args.traces.resolve() / "manifest.json", OUT_DIR=args.out.resolve())
notebook = json.loads(Path("Step 2 Calibration Integrated.ipynb").read_text(encoding="utf-8"))
environment = dict(display=print)
applied = set()
for idx, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue
    tree = ast.parse("".join(cell["source"]))
    if not applied:
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in settings and name not in applied:
                    value = settings[name]
                    node.value = ast.parse(f"Path({str(value)!r})" if isinstance(value, Path)
                                           else repr(value), mode="eval").body
                    applied.add(name)
        if applied != set(settings):
            raise RuntimeError("Could not bind real-run notebook inputs")
    print(f"Executing notebook cell {idx}", flush=True)
    exec(compile(ast.fix_missing_locations(tree), f"<notebook cell {idx}>", "exec"), environment)
