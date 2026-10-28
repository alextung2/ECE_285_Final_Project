"""Shared CLI plumbing for the numbered pipeline scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))   # so scripts work without `pip install -e .`

from mcal.config import load_config  # noqa: E402


def parse(description: str, extra=None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. traces.topk=10")
    ap.add_argument("--synthetic", action="store_true",
                    help="use synthetic traces (outputs go to <out_dir>/synthetic)")
    if extra:
        extra(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.synthetic:
        cfg["out_dir"] = str(Path(cfg["out_dir"]) / "synthetic")
    cfg["out_dir"] = str((ROOT / cfg["out_dir"]).resolve()) if not Path(cfg["out_dir"]).is_absolute() else cfg["out_dir"]
    return args, cfg


def write_table(rows: list[dict], path: Path, cols: list[str] | None = None) -> str:
    """Write rows as CSV and return a markdown version (for pasting into the report)."""
    import csv
    cols = cols or list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    md = "\n".join(md)
    path.with_suffix(".md").write_text(md + "\n")
    return md
