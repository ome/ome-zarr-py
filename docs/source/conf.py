import pathlib
import sys

# alternative is to make code installable (which it is!)
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())


extensions = [
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
]

# use index.rst instead of contents.rst
master_doc = "index"
