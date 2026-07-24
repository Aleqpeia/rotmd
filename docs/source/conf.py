"""Sphinx configuration for the rotmd docs site."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make `rotmd` importable without installing the package, and expose the
# version string autodoc/autosummary need at import time.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import rotmd  # noqa: E402

project = "rotmd"
author = "Mykyta Bobylyow"
copyright = f"2026, {author}"
release = getattr(rotmd, "__version__", "0.1.0")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # safety net: rotmd.base predates the current docstring style
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",  # analysis/*.py docstrings use :math: roles
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "both"

# Modules imported at module scope by rotmd but not part of the default
# install (e.g. the commented-out `gpu` poetry group). Add here, not by
# installing torch in CI, if autodoc import errors show up for a module.
autodoc_mock_imports = [
    "torch",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- HTML output --------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = []

# -- MyST ----------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
