"""Shared prompt splits for Ada's notebook and the CLI pipeline."""
import json
from pathlib import Path
import numpy as np


def split_calibration(traces, manifest=None, fraction=.8, seed=0):
    if manifest:
        data = json.loads(Path(manifest).read_text(encoding='utf-8'))
        train, val = data['train_prompt_ids'], data['validation_prompt_ids']
        for ids in (train, val):
            if not ids or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
                raise ValueError('Manifest requires nonempty unique integer prompt IDs')
        if set(train) & set(val):
            raise ValueError('Manifest train/validation overlap')
        if set(train) | set(val) != set(traces.prompt_id.tolist()):
            raise ValueError('Manifest must exactly cover the collected prompt IDs; no silent fallback')
        fit, held = traces.split_by_prompt_ids(train)
    else:
        if traces.meta.get('source') != 'synthetic':
            raise ValueError('Real calibration requires a fixed train/validation manifest')
        if not 0 < fraction < 1:
            raise ValueError('Training fraction must be between zero and one')
        fit, held = traces.split_by_prompt(fraction, seed)
    if not fit.n_steps or not held.n_steps:
        raise ValueError('Both calibration partitions must be nonempty')
    return fit, held
