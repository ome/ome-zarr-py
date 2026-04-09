import pathlib
import sys

# alternative is to make code installable (which it is!)
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

exclude_patterns = ["**.ipynb_checkpoints", ".DS_Store", "Thumbs.db", "_build"]
extensions = [
    "sphinx_togglebutton",
    "sphinx_copybutton",
    "myst_nb",
    "sphinx_thebe",
    "sphinx_comments",
    "sphinx_external_toc",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_book_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    # "sphinxcontrib.bibtex",
    "sphinx_jupyterbook_latex",
]

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "linkify",
    "substitution",
    "tasklist",
]

# -- Project information -----------------------------------------------------

project = "ome-zarr-py"
copyright = "Open Microscopy Environment"  # noqa: A001
author = "Open Microscopy Environment"

# Example configuration for intersphinx: refer to the Python standard library.
# use in refs e.g:
# :ref:`comparison manual <python:comparisons>`
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable/", None),
    "dask": ("https://docs.dask.org/en/stable", None),
}

# https://github.com/readthedocs/sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"
