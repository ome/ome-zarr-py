# Expose __version__ and fallback when _version.py doesn't exist.

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0+unknown"


__all__ = ["__version__"]
