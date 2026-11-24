"""Prompt identity and collection completeness, independent of GPU label truth."""
import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np


def read_prompt_records(path):
    records = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    validate_prompt_records(records)
    return records


def validate_prompt_records(records):
    if not isinstance(records, list) or not records:
        raise ValueError("Prompt records must be a nonempty list")
    ids = []
    for r in records:
        if (not isinstance(r, dict) or type(r.get("prompt_id")) is not int
                or r["prompt_id"] < 0 or not isinstance(r.get("instruction"), str)
                or not r["instruction"].strip()):
            raise ValueError("Each prompt needs a nonnegative integer prompt_id and nonempty instruction")
        ids.append(r["prompt_id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate prompt IDs")


def prompt_digest(records):
    """Preserve the original collector's ordered ID/text hash algorithm."""
    validate_prompt_records(records)
    pairs = [(r["prompt_id"], r["instruction"]) for r in records]
    return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()


def normalized_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def require_disjoint_text(left, right):
    validate_prompt_records(left)
    validate_prompt_records(right)
    a = {normalized_text(r["instruction"]) for r in left}
    b = {normalized_text(r["instruction"]) for r in right}
    if a & b:
        raise ValueError(f"Prompt text overlap across splits: {len(a & b)} normalized texts")
    return "No exact or Unicode/case/whitespace-normalized text overlap; semantic overlap not assessed"


def exclude_evaluation_overlap(records, evaluation):
    """Prepare new calibration inputs; never edit existing traces or evaluation."""
    validate_prompt_records(records)
    validate_prompt_records(evaluation)
    lookup = {}
    for record in evaluation:
        lookup.setdefault(normalized_text(record["instruction"]), []).append(record["prompt_id"])
    kept, exclusions = [], []
    for record in records:
        matched = lookup.get(normalized_text(record["instruction"]))
        if matched:
            exclusions.append(dict(prompt_id=record["prompt_id"], evaluation_prompt_ids=matched,
                                   instruction=record["instruction"], reason="evaluation text overlap"))
        else:
            kept.append(record)
    if not kept:
        raise ValueError("No calibration prompts remain after evaluation-overlap exclusions")
    return kept, exclusions


def check_prompt_binding(traces, records):
    validate_prompt_records(records)
    expected = {r["prompt_id"] for r in records}
    present = set(np.unique(traces.prompt_id).tolist())
    if present != expected:
        raise ValueError(f"Prompt coverage mismatch: missing={sorted(expected-present)}, extra={sorted(present-expected)}")
    if traces.meta.get("prompt_sha256") != prompt_digest(records):
        raise ValueError("Prompt ID/text hash does not match trace metadata")
    return dict(prompts=len(expected), prompt_sha256=prompt_digest(records), exact_coverage=True)


def check_collection_status(traces):
    """Reject explicit partial collections; historical absence is not certification."""
    if traces.meta.get("collection_complete") is False or traces.meta.get("missing_prompt_ids"):
        raise ValueError("Incomplete collection: preserve it for diagnosis, do not fit final models")
    expected = traces.meta.get("expected_prompt_ids")
    if expected is not None:
        if (not isinstance(expected, list) or not expected
                or any(type(i) is not int or i < 0 for i in expected)
                or len(set(expected)) != len(expected)):
            raise ValueError("Invalid expected_prompt_ids")
        if set(expected) != set(np.unique(traces.prompt_id).tolist()):
            raise ValueError("Incomplete collection: expected prompt IDs differ from stored rows")
