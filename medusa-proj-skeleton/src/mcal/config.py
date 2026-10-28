from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path = "configs/default.yaml", overrides: list[str] | None = None) -> dict:
    """Load YAML config; overrides like ["traces.topk=10", "seed=1"]."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        d = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            d = d.setdefault(k, {})
        d[leaf] = yaml.safe_load(val)
    return cfg


def out_path(cfg: dict, *parts: str) -> Path:
    p = Path(cfg["out_dir"]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
