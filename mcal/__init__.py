"""Repository-root imports use the canonical pipeline package.

The sibling files are preserved Ada source snapshots, not the active implementation.
Use Step 2 Calibration Integrated.ipynb; the original notebook is historical.
"""
from pathlib import Path
__path__ = [str(Path(__file__).resolve().parents[1] / "medusa-proj-skeleton" / "src" / "mcal")]
