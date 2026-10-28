"""Prompt sets: AlpacaEval (calibration) and MT-Bench (evaluation)."""

from __future__ import annotations

import json
from pathlib import Path

VICUNA_SYSTEM = ("A chat between a curious user and an artificial intelligence assistant. "
                 "The assistant gives helpful, detailed, and polite answers to the user's questions.")


def vicuna_prompt(user_msg: str) -> str:
    """Vicuna v1.1/v1.3 conversation template (single turn)."""
    return f"{VICUNA_SYSTEM} USER: {user_msg} ASSISTANT:"


def load_alpacaeval(n: int | None = None) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval", trust_remote_code=True)
    prompts = [ex["instruction"] for ex in ds]
    return prompts[:n] if n else prompts


def load_mtbench(path: str | Path, n: int | None = None, first_turn_only: bool = True) -> list[str]:
    """MT-Bench questions in FastChat format (fastchat/llm_judge/data/mt_bench/question.jsonl).

    Download once into data/mt_bench_question.jsonl. Only the first turn is used
    by default; multi-turn needs the previous answer in context (TODO if wanted).
    """
    prompts = []
    with open(path) as f:
        for line in f:
            q = json.loads(line)
            prompts.append(q["turns"][0] if first_turn_only else q["turns"])
    return prompts[:n] if n else prompts


def load_humaneval(n: int | None = None) -> list[str]:
    """Optional (proposal Sec. 5)."""
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    prompts = [ex["prompt"] for ex in ds]
    return prompts[:n] if n else prompts


def load_prompts(split_cfg: dict) -> list[str]:
    name, n = split_cfg["dataset"], split_cfg.get("n_prompts")
    if name == "alpacaeval":
        return load_alpacaeval(n)
    if name == "mtbench":
        return load_mtbench(split_cfg["path"], n)
    if name == "humaneval":
        return load_humaneval(n)
    raise ValueError(name)
