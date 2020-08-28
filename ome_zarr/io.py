"""Reading logic for ome-zarr.

Primary entry point is the :func:`~ome_zarr.io.parse_url` method.
"""

import json
import logging
import os
import posixpath
from abc import ABC, abstractmethod
from typing import Optional
from urllib.parse import urlparse

import dask.array as da
import requests

from .types import JSONDict

LOGGER = logging.getLogger("ome_zarr.io")


class BaseZarrLocation(ABC):
    """
    Base IO primitive for reading Zarr data.

    No assumptions about the existence of the given path string are made.
    Attempts are made to load various metadata files and cache them internally.
    """

    def __init__(self, path: str) -> None:
        self.zarr_path: str = path.endswith("/") and path or f"{path}/"
        self.zarray: JSONDict = self.get_json(".zarray")
        self.zgroup: JSONDict = self.get_json(".zgroup")
        self.__metadata: JSONDict = {}
        self.__exists: bool = True
        if self.zgroup:
            self.__metadata = self.get_json(".zattrs")
        elif self.zarray:
            self.__metadata = self.get_json(".zattrs")
        else:
            self.__exists = False

    def __repr__(self) -> str:
        """Print the path as well as whether this is a group or an array."""
        suffix = ""
        if self.zgroup:
            suffix += " [zgroup]"
        if self.zarray:
            suffix += " [zarray]"
        return f"{self.zarr_path}{suffix}"

    def exists(self) -> bool:
        """Return true if zgroup or zarray metadata exists."""
        return self.__exists

    def is_zarr(self) -> Optional[JSONDict]:
        """Return true if either zarray or zgroup metadata exists."""
        return self.zarray or self.zgroup

    @property
    def root_attrs(self) -> JSONDict:
        """Return the contents of the zattrs file."""
        return dict(self.__metadata)

    @abstractmethod
    def get_json(self, subpath: str) -> JSONDict:
        """Must be implemented by subclasses."""
        raise NotImplementedError("unknown")

    def load(self, subpath: str) -> da.core.Array:
        """Use dask.array.from_zarr to load the subpath."""
        return da.from_zarr(f"{self.zarr_path}{subpath}")

    # TODO: update to from __future__ import annotations with 3.7+
    def create(self, path: str) -> "BaseZarrLocation":
        """Create a new Zarr location for the given path."""
        subpath = posixpath.join(self.zarr_path, path)
        subpath = posixpath.normpath(subpath)
        LOGGER.debug(f"open({self.__class__.__name__}({subpath}))")
        return self.__class__(posixpath.normpath(f"{subpath}"))


class LocalZarrLocation(BaseZarrLocation):
    """
    Uses the :module:`json` library for loading JSON from disk.
    """

    def get_json(self, subpath: str) -> JSONDict:
        """
        Load and return a given subpath of self.zarr_path as JSON.

        If a file does not exist, an empty response is returned rather
        than an exception.
        """
        filename = os.path.join(self.zarr_path, subpath)

        if not os.path.exists(filename):
            LOGGER.debug(f"{filename} does not exist")
            return {}

        with open(filename) as f:
            return json.loads(f.read())


class RemoteZarrLocation(BaseZarrLocation):
    """ Uses the :module:`requests` library for accessing Zarr metadata files. """

    def get_json(self, subpath: str) -> JSONDict:
        """
        Load and return a given subpath of self.zarr_path as JSON.

        HTTP 403 and 404 responses are treated as if the file does not exist.
        Exceptions during the remote connection are logged at the WARN level.
        All other exceptions log at the ERROR level.
        """
        url = f"{self.zarr_path}{subpath}"
        try:
            rsp = requests.get(url)
        except Exception:
            LOGGER.warn(f"unreachable: {url} -- details logged at debug")
            LOGGER.debug("exception details:", exc_info=True)
            return {}
        try:
            if rsp.status_code in (403, 404):  # file doesn't exist
                return {}
            return rsp.json()
        except Exception:
            LOGGER.error(f"({rsp.status_code}): {rsp.text}")
            return {}


def parse_url(path: str) -> Optional[BaseZarrLocation]:
    """Convert a path string or URL to a BaseZarrLocation subclass.

    >>> parse_url('does-not-exist')
    """
    # Check is path is local directory first
    if os.path.isdir(path):
        return LocalZarrLocation(path)
    else:
        result = urlparse(path)
        zarr: Optional[BaseZarrLocation] = None
        if result.scheme in ("", "file"):
            # Strips 'file://' if necessary
            zarr = LocalZarrLocation(result.path)
        else:
            zarr = RemoteZarrLocation(path)
        if zarr.is_zarr():
            return zarr
    return None
