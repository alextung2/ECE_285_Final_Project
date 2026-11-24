"""Identical policy loading for untimed correctness and latency experiments."""
import json
from pathlib import Path

from .decode import DynamicTree, StaticTree
from .estimators import Scorer, MedusaPlusProb
from .precision import require_compatible


def load_policy(name, budget, contract, model_dir):
    if name == "medusa_default":
        from .collect import default_medusa_choices
        return StaticTree(default_medusa_choices())
    if name == "medusa_plus":
        return DynamicTree(MedusaPlusProb(), budget)
    model_dir = Path(model_dir)
    if name in ("medusa_acc", "count_cond"):
        path = model_dir / f"fixed_tree_{name}_B{budget}.json"
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"Legacy tree lacks execution contract: {path}")
        require_compatible(contract, payload.get("execution_contract"))
        policy = StaticTree(payload["choices"])
        if len(policy.choices) != budget:
            raise ValueError(f"{path} does not contain {budget} draft nodes")
        return policy
    if name in ("direct", "logistic_cond"):
        scorer = Scorer.load(model_dir / f"{name}.pkl")
        require_compatible(contract, getattr(scorer, "execution_contract", None))
        return DynamicTree(scorer, budget)
    raise ValueError(f"Unknown policy: {name}")
