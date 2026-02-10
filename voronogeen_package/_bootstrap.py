"""Helpers for importing the legacy ``vorongen`` package from source."""

from pathlib import Path
import sys


def ensure_vorongen_on_path() -> None:
    """Add the local ``src`` folder to ``sys.path`` when available."""
    root = Path(__file__).resolve().parents[1]
    src_dir = root / "src"
    src_text = str(src_dir)
    if src_dir.is_dir() and src_text not in sys.path:
        sys.path.insert(0, src_text)
