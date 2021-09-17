"""Reading logic for ome-zarr.

Primary entry point is the :func:`~ome_zarr.io.parse_url` method.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urljoin

import dask.array as da
from zarr.storage import FSStore

from .format import CurrentFormat, Format, detect_format
from .types import JSONDict

LOGGER = logging.getLogger("ome_zarr.io")


class ZarrLocation:
    """
    IO primitive for reading and writing Zarr data. Uses FSStore for all
    data access.

    No assumptions about the existence of the given path string are made.
    Attempts are made to load various metadata files and cache them internally.
    """

    def __init__(
        self, path: Union[Path, str], mode: str = "r", fmt: Format = CurrentFormat()
    ) -> None:

        self.__fmt = fmt
        self.__mode = mode
        if isinstance(path, Path):
            self.__path = str(path.resolve())
        elif isinstance(path, str):
            self.__path = path
        else:
            raise TypeError(f"not expecting: {type(path)}")

        loader = fmt
        if loader is None:
            loader = CurrentFormat()
        self.__store = loader.init_store(self.__path, mode)

        self.__init_metadata()
        detected = detect_format(self.__metadata)
        if detected != fmt:
            LOGGER.warning(f"version mismatch: detected:{detected}, requested:{fmt}")
            self.__fmt = detected
            self.__store = detected.init_store(self.__path, mode)
            self.__init_metadata()

    def __init_metadata(self) -> None:
        """
        Load the Zarr metadata files for the given location.
        """
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
    def fmt(self) -> Format:
        return self.__fmt

    @property
    def mode(self) -> str:
        return self.__mode

    @property
    def path(self) -> str:
        return self.__path

    @property
    def store(self) -> FSStore:
        """Return the initialized store for this location"""
        assert self.__store is not None
        return self.__store

    @property
    def root_attrs(self) -> JSONDict:
        """Return the contents of the zattrs file."""
        return dict(self.__metadata)

    def load(self, subpath: str = "") -> da.core.Array:
        """Use dask.array.from_zarr to load the subpath."""
        return da.from_zarr(self.__store, subpath)

    def __eq__(self, rhs: object) -> bool:
        if type(self) != type(rhs):
            return False
        if not isinstance(rhs, ZarrLocation):
            return False
        return self.subpath() == rhs.subpath()

    def basename(self) -> str:
        """Return the last element of the underlying location.

        >>> ZarrLocation("/tmp/foo").basename()
        'foo'
        >>> ZarrLocation("https://example.com/bar").basename()
        'bar'
        >>> ZarrLocation("https://example.com/baz/").basename()
        'baz'
        """
        path = self.__path.endswith("/") and self.__path[0:-1] or self.__path
        return path.split("/")[-1]

    # TODO: update to from __future__ import annotations with 3.7+
    def create(self, path: str) -> "ZarrLocation":
        """Create a new Zarr location for the given path."""
        subpath = self.subpath(path)
        LOGGER.debug(f"open({self.__class__.__name__}({subpath}))")
        return self.__class__(subpath, mode=self.__mode, fmt=self.__fmt)

    def get_json(self, subpath: str) -> JSONDict:
        """
        Load and return a given subpath of store as JSON.

        HTTP 403 and 404 responses are treated as if the file does not exist.
        Exceptions during the remote connection are logged at the WARN level.
        All other exceptions log at the ERROR level.
        """
        try:
            data = self.__store.get(subpath)
            if not data:
                return {}
            return json.loads(data)
        except KeyError:
            LOGGER.debug(f"JSON not found: {subpath}")
            return {}
        except Exception as e:
            LOGGER.exception(f"{e}")
            return {}

    def parts(self) -> List[str]:
        if self.__store.fs.protocol == "file":
            return list(Path(self.__path).parts)
        else:
            return self.__path.split("/")

    def subpath(self, subpath: str = "") -> str:
        if self.__store.fs.protocol == "file":
            filename = Path(self.__path) / subpath
            filename = filename.resolve()
            return str(filename)
        else:
            url = str(self.__path)
            if not url.endswith("/"):
                url = f"{url}/"
            return urljoin(url, subpath)


def parse_url(
    path: Union[Path, str], mode: str = "r", fmt: Format = CurrentFormat()
) -> Optional[ZarrLocation]:
    """Convert a path string or URL to a ZarrLocation subclass.

    >>> parse_url('does-not-exist')
    """
    try:
        loc = ZarrLocation(path, mode=mode, fmt=fmt)
        if "r" in mode and not loc.exists():
            return None
        else:
            return loc
    except Exception as e:
        LOGGER.warning(f"exception on parsing: {e} (stacktrace at DEBUG)")
        LOGGER.debug("stacktrace:", exc_info=True)
        return None
