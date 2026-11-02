"""Read-only, CPU-only trace handoff checks. Does not certify GPU labels."""
from pathlib import Path
import hashlib
import numpy as np
from .precision import require_trace_config, require_compatible, trace_contract
from .traces import TraceSet


def validate_arrays(ts):
    if not isinstance(ts.meta, dict):
        raise ValueError('Trace metadata must be a JSON object')
    if ts.gold_rank.ndim != 2 or ts.topk_prob.ndim != 3:
        raise ValueError('Expected gold_rank [N,H] and topk_prob [N,H,K]')
    n, h = ts.gold_rank.shape
    if not n or not h or ts.topk_prob.shape[:2] != (n, h) or ts.topk < 1:
        raise ValueError('Empty or inconsistent trace dimensions')
    for name in ('root_prob', 'prompt_id', 'position'):
        if getattr(ts, name).shape != (n,):
            raise ValueError(f'{name} must have shape [N]')
    for name in ('gold_rank', 'prompt_id', 'position'):
        if not np.issubdtype(getattr(ts, name).dtype, np.integer):
            raise ValueError(f'{name} must contain integers, not inferred/rounded labels')
    if np.any((ts.gold_rank < -1) | (ts.gold_rank >= ts.topk)):
        raise ValueError('gold_rank must be -1 or a zero-based top-K rank')
    if np.any(ts.position < 0) or np.any(ts.prompt_id < 0):
        raise ValueError('Negative prompt ID or position')
    pairs = np.column_stack((ts.prompt_id, ts.position))
    if len(np.unique(pairs, axis=0)) != n:
        raise ValueError('Duplicate prompt_id/position states')
    for name in ('topk_prob', 'root_prob'):
        p = getattr(ts, name)
        if not np.issubdtype(p.dtype, np.floating) or not np.isfinite(p).all():
            raise ValueError(f'{name} must contain finite floating probabilities')
        if np.any((p < 0) | (p > 1)):
            raise ValueError(f'{name} outside [0,1]')
    if np.any(ts.topk_prob.sum(-1) > 1 + 1e-5):
        raise ValueError('Top-K probability mass exceeds one; do not renormalize missing entries')
    if ts.meta.get('source') != 'synthetic' and np.any(np.diff(ts.topk_prob, axis=-1) > 1e-7):
        raise ValueError('Real top-K probabilities are not in descending candidate-rank order')
    if ts.topk_token is not None:
        if ts.topk_token.shape != ts.topk_prob.shape or not np.issubdtype(ts.topk_token.dtype, np.integer):
            raise ValueError('topk_token must be integer [N,H,K]')
        if np.any(ts.topk_token < 0) or np.any(np.diff(np.sort(ts.topk_token, axis=-1), axis=-1) == 0):
            raise ValueError('Negative or duplicate top-K token IDs')


def audit_file(path, cfg, synthetic=False):
    path = Path(path)
    ts = TraceSet.load(path)
    validate_arrays(ts)
    contract = require_trace_config(ts, cfg, synthetic)
    if ts.topk != cfg['traces']['topk']:
        raise ValueError('Trace top-K differs from configuration')
    warnings = []
    if not synthetic:
        for key in ('packages', 'medusa_source_sha256', 'loaded_heads_sha256', 'base_revision'):
            if not contract.get(key):
                raise ValueError(f'Missing runtime provenance: {key}')
        if not ts.meta.get('dataset'):
            warnings.append('Dataset identity missing: split provenance cannot be checked')
        if not ts.meta.get('dataset_revision'):
            warnings.append('Dataset revision absent: obtain original collection manifest')
        if ts.topk_token is None:
            warnings.append('Token IDs absent: ranks cannot be replayed against original candidates')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return ts, dict(path=str(path.resolve()), sha256=digest.hexdigest(), steps=ts.n_steps,
                    prompts=len(np.unique(ts.prompt_id)), n_heads=ts.n_heads, topk=ts.topk,
                    source=ts.meta.get('source'), execution_contract=contract, warnings=warnings)


def audit_pair(calib, evaluation):
    require_compatible(trace_contract(calib), trace_contract(evaluation))
    # Dataset-local integers cannot establish cross-dataset prompt identity.
    dataset = calib.meta.get('dataset')
    if dataset and dataset == evaluation.meta.get('dataset'):
        overlap = np.intersect1d(calib.prompt_id, evaluation.prompt_id)
        if overlap.size:
            raise ValueError(f'Calibration/evaluation prompt IDs overlap in {dataset}: {overlap.tolist()}')
        if calib.meta.get('dataset_revision') != evaluation.meta.get('dataset_revision'):
            raise ValueError('Same-dataset splits have different revisions; verify the original manifests')
        return 'Disjoint IDs in the named dataset; original manifests still needed to verify text identity'
    return 'Different/unknown datasets: numeric IDs are not comparable; check source manifests for text overlap'
