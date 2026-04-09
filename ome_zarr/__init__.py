from dask import __version__ as dask_version
from packaging.version import Version

# Expose __version__ and fallback when _version.py doesn't exist.
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0+unknown"

from .classes import NgffImage, NgffMultiscales

# If not 2026.3.0 it must be 2025.11.0 or lower. Name indicates kwargs only contain array kwargs in the dask version.
USE_DASK_ARRAY_KWARGS = Version(dask_version) >= Version("2026.3.0")

__all__ = ["NgffImage", "NgffMultiscales", "__version__"]
