"""Sphinx configuration for vorongen documentation."""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
_version_match = re.search(r'^version\s*=\s*"([^"]+)"', _pyproject_text, re.MULTILINE)
_release = _version_match.group(1) if _version_match else "0.0.0"

project = "vorongen"
author = "Vorongen Contributors"
copyright = f"{datetime.now().year}, {author}"
release = _release


extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"

napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = ["colon_fence", "deflist"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}


html_theme = "furo"
html_static_path = []
