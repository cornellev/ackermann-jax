import sys
import os

sys.path.insert(0, os.path.abspath("../src"))

project = "ackermann-jax"
author = "Lucas Libshutz"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_typehints = "signature"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/lucaslibshutz/ackermann-jax",
    "show_toc_level": 2,
    "navigation_with_keys": True,
}

templates_path = ["_templates"]
exclude_patterns = ["_build"]
