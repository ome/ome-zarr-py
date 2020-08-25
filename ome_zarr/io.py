"""
Reading logic for ome-zarr
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
    def __init__(self, path: str) -> None:
        self.zarr_path: str = path.endswith("/") and path or f"{path}/"
        self.zarray: JSONDict = self.get_json(".zarray")
        self.zgroup: JSONDict = self.get_json(".zgroup")
        self.root_attrs: JSONDict = {}
        if self.zgroup:
            self.root_attrs = self.get_json(".zattrs")
        elif self.zarray:
            self.root_attrs = self.get_json(".zattrs")

    def __repr__(self) -> str:
        suffix = ""
        if self.zgroup:
            suffix += " [zgroup]"
        if self.zarray:
            suffix += " [zarray]"
        return f"{self.zarr_path}{suffix}"

    def exists(self) -> bool:
        return os.path.exists(self.zarr_path)

    def is_zarr(self) -> Optional[JSONDict]:
        return self.zarray or self.zgroup

    @abstractmethod
    def get_json(self, subpath: str) -> JSONDict:
        raise NotImplementedError("unknown")

    def load(self, subpath: str) -> da.core.Array:
        """
        Use dask.array.from_zarr to load the subpath
        """
        return da.from_zarr(f"{self.zarr_path}{subpath}")

    # TODO: update to from __future__ import annotations with 3.7+
    def open(self, path: str) -> "BaseZarrLocation":
        """Create a new zarr for the given path"""
        subpath = posixpath.join(self.zarr_path, path)
        subpath = posixpath.normpath(subpath)
        LOGGER.debug(f"open({self.__class__.__name__}({subpath}))")
        return self.__class__(posixpath.normpath(f"{subpath}"))


class LocalZarrLocation(BaseZarrLocation):
    def get_json(self, subpath: str) -> JSONDict:
        filename = os.path.join(self.zarr_path, subpath)

        if not os.path.exists(filename):
            return {}

        with open(filename) as f:
            return json.loads(f.read())


class RemoteZarrLocation(BaseZarrLocation):
    def get_json(self, subpath: str) -> JSONDict:
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
    """ convert a path string or URL to a BaseZarrLocation instance
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
