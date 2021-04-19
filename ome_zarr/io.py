"""Reading logic for ome-zarr.

Primary entry point is the :func:`~ome_zarr.io.parse_url` method.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import dask.array as da
from zarr.core import Array
from zarr.storage import FSStore

from .types import JSONDict

LOGGER = logging.getLogger("ome_zarr.io")


class BaseZarrLocation(ABC):
    """
    Base IO primitive for reading Zarr data.

    No assumptions about the existence of the given path string are made.
    Attempts are made to load various metadata files and cache them internally.
    """

    def __init__(self) -> None:
        self._store: Optional[FSStore] = None

    def _load(self) -> None:
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
        return f"{self.subpath('')}{suffix}"

    def exists(self) -> bool:
        """Return true if either zgroup or zarray metadata exists."""
        return self.__exists

    @property
    def store(self) -> FSStore:
        """Return the initialized store for this location"""
        assert self._store is not None
        return self._store

    @property
    def root_attrs(self) -> JSONDict:
        """Return the contents of the zattrs file."""
        return dict(self.__metadata)

    def load(self, subpath: str = "") -> da.core.Array:
        """Use dask.array.from_zarr to load the subpath."""
        return da.from_zarr(f"{self.subpath(subpath)}")

    def __eq__(self, rhs: object) -> bool:
        if type(self) != type(rhs):
            return False
        if not isinstance(rhs, BaseZarrLocation):
            return False
        return self.subpath() == rhs.subpath()

    # Abstract methods to be implemented by subclasses

    @abstractmethod
    def basename(self) -> str:
        """Return the last element of the underlying location.

        >>> RemoteZarrLocation("https://example.com/foo").basename()
        'foo'
        """
        raise NotImplementedError("unknown")

    # TODO: update to from __future__ import annotations with 3.7+
    @abstractmethod
    def create(self, path: str) -> "BaseZarrLocation":
        """Must be implemented by subclasses."""
        raise NotImplementedError("unknown")

    @abstractmethod
    def get_json(self, subpath: str) -> JSONDict:
        """Must be implemented by subclasses."""
        raise NotImplementedError("unknown")

    @abstractmethod
    def parts(self) -> List[str]:
        """Must be implemented by subclasses."""
        raise NotImplementedError("unknown")

    @abstractmethod
    def subpath(self, subpath: str = "") -> str:
        """Must be implemented by subclasses."""
        raise NotImplementedError("unknown")


class LocalZarrLocation(BaseZarrLocation):
    """
    Uses the :module:`json` library for loading JSON from disk.
    """

    def __init__(self, path: Path, mode: str = "r") -> None:
        super().__init__()
        self.__path: Path = path
        self.mode = mode
        self._store = FSStore(
            str(self.__path),  # TODO: open issue for using Path
            auto_mkdir=True,
            key_separator="/",
            mode=mode,
        )
        LOGGER.debug("Created FSStore %s", self.basename())
        self._load()

    def basename(self) -> str:
        return self.__path.name

    def subpath(self, subpath: str = "") -> str:
        return str((self.__path / subpath).resolve())

    def parts(self) -> List[str]:
        return list(self.__path.parts)

    def create(self, path: str) -> "LocalZarrLocation":
        """Create a new Zarr location for the given path."""
        subpath = (self.__path / path).resolve()
        LOGGER.debug(f"open({self.__class__.__name__}({subpath}))")
        return self.__class__(subpath)

    def get_json(self, subpath: str) -> JSONDict:
        """
        Load and return a given subpath of self.__path as JSON.

        If a file does not exist, an empty response is returned rather
        than an exception.
        """
        filename = self.subpath(subpath)
        if not os.path.exists(filename):
            LOGGER.debug(f"{filename} does not exist")
            return {}

        with open(filename) as f:
            return json.loads(f.read())


class RemoteZarrLocation(BaseZarrLocation):
    """ Uses the :module:`requests` library for accessing Zarr metadata files. """

    def __init__(self, url: str) -> None:
        super().__init__()
        self.__url: str = url.endswith("/") and url or f"{url}/"
        self._store = FSStore(
            url, key_separator="/", mode="r"
        )  # FIXME: allow mode "w"?
        LOGGER.debug("Created read-only FSStore %s", self.basename())
        self._load()

    def basename(self) -> str:
        url = self.__url.endswith("/") and self.__url[0:-1] or self.__url
        return url.split("/")[-1]

    def subpath(self, path: str = "") -> str:
        return urljoin(self.__url, path)

    def parts(self) -> List[str]:
        return self.__url.split("/")

    def create(self, path: str) -> "RemoteZarrLocation":
        """Create a new Zarr location for the given path."""
        subpath = self.subpath(path)
        LOGGER.debug(f"open({self.__class__.__name__}({subpath}))")
        return self.__class__(f"{subpath}")

    def load(self, subpath: str = "") -> da.core.Array:
        """Use dask.array.from_zarr to load the subpath."""
        # TODO: remove base implementation?
        array = Array(self.store, subpath)
        return da.from_zarr(array)

    def get_json(self, subpath: str) -> JSONDict:
        """
        Load and return a given subpath of self.__url as JSON.

        HTTP 403 and 404 responses are treated as if the file does not exist.
        Exceptions during the remote connection are logged at the WARN level.
        All other exceptions log at the ERROR level.
        """
        try:
            return json.loads(self.store.get(subpath))
        except KeyError:
            return {}
        except Exception as e:
            LOGGER.error(f"{e}")
            return {}


def parse_url(path: str, mode: str = "r") -> Optional[BaseZarrLocation]:
    """Convert a path string or URL to a BaseZarrLocation subclass.

    >>> parse_url('does-not-exist')
    """
    # Check is path is local directory first
    if os.path.isdir(path):
        LOGGER.debug(f"returning local directory {path}")
        return LocalZarrLocation(Path(path), mode=mode)
    else:
        result = urlparse(path)
        zarr_loc: Optional[BaseZarrLocation] = None
        if result.scheme in ("", "file"):
            # Strips 'file://' if necessary
            zarr_loc = LocalZarrLocation(Path(result.path), mode=mode)
            LOGGER.debug(f"found local uri {path}")

        else:
            if mode != "r":
                raise ValueError("Remote locations are read only")
            zarr_loc = RemoteZarrLocation(path)
            LOGGER.debug(f"found remote uri {path}")
        if zarr_loc.exists() or (mode in ("a", "w")):
            LOGGER.debug(f"uri exists or will be created: {path}")
            return zarr_loc
    LOGGER.debug(f"no location parsed from {path}")
    return None
