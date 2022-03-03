# add conf when needed

# import pathlib
# import sys

# print("conf.py parents", pathlib.Path(__file__).parents)
# alternative is to make code installable (which it is!)
# sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

# for p in sys.path:
#     print(p)

extensions = [
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
]
