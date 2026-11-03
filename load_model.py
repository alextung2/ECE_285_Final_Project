"""Shared Colab loader. Run from a complete checkout; keep old Drive runs intact."""
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "medusa-proj-skeleton" / "src"))
from mcal.config import load_config
from mcal.model import load_medusa

cfg = load_config(root / "medusa-proj-skeleton" / "configs" / "default.yaml")
cfg["model"]["precision_mode"] = os.environ.get("MEDUSA_PRECISION_MODE", "nf4_fp16")
cfg["model"]["device_map"] = {"": 0}
model_id = cfg["model"]["name"]
model, tokenizer = load_medusa(cfg)
print("Execution contract:", model.execution_contract)
print("Runtime dtype audit:", model.precision_audit)
