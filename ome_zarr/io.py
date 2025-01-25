"""Reading logic for ome-zarr.

Primary entry point is the :func:`~ome_zarr.io.parse_url` method.
"""

import json
import logging
from pathlib import Path
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
        self,
        path: Path | str | FSStore,
        mode: str = "r",
        fmt: Format = CurrentFormat(),
    ) -> None:
        LOGGER.debug("ZarrLocation.__init__ path: %s, fmt: %s", path, fmt.version)
        self.__fmt = fmt
        self.__mode = mode
        if isinstance(path, Path):
            self.__path = str(path.resolve())
        elif isinstance(path, str):
            self.__path = path
        elif isinstance(path, FSStore):
            self.__path = path.path
        else:
            raise TypeError(f"not expecting: {type(path)}")

        loader = fmt
        if loader is None:
            loader = CurrentFormat()
        self.__store: FSStore = (
            path if isinstance(path, FSStore) else loader.init_store(self.__path, mode)
        )

        self.__init_metadata()
        detected = detect_format(self.__metadata, loader)
        LOGGER.debug("ZarrLocation.__init__ %s detected: %s", path, detected)
        if detected != fmt:
            LOGGER.warning(
                "version mismatch: detected: %s, requested: %s", detected, fmt
            )
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
        if type(self) is not type(rhs):
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
        path = (self.__path.endswith("/") and self.__path[0:-1]) or self.__path
        return path.split("/")[-1]

    # TODO: update to from __future__ import annotations with 3.7+
    def create(self, path: str) -> "ZarrLocation":
        """Create a new Zarr location for the given path."""
        subpath = self.subpath(path)
        LOGGER.debug("open(%s(%s))", self.__class__.__name__, subpath)
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
            LOGGER.debug("JSON not found: %s", subpath)
            return {}
        except Exception:
            LOGGER.exception("Error while loading JSON")
            return {}

    def parts(self) -> list[str]:
        if self._isfile():
            return list(Path(self.__path).parts)
        else:
            return self.__path.split("/")

    def subpath(self, subpath: str = "") -> str:
        if self._isfile():
            filename = Path(self.__path) / subpath
            filename = filename.resolve()
            return str(filename)
        elif self._ishttp():
            url = str(self.__path)
            if not url.endswith("/"):
                url = f"{url}/"
            return urljoin(url, subpath)
        else:
            # Might require a warning
            if self.__path.endswith("/"):
                return f"{self.__path}{subpath}"
            else:
                return f"{self.__path}/{subpath}"

    def _isfile(self) -> bool:
        """
        Return whether the current underlying implementation
        points to a local file or not.
        """
        return self.__store.fs.protocol == "file" or self.__store.fs.protocol == (
            "file",
            "local",
        )

    def _ishttp(self) -> bool:
        """
        Return whether the current underlying implementation
        points to a URL
        """
        return self.__store.fs.protocol in ["http", "https"]


def parse_url(
    path: Path | str, mode: str = "r", fmt: Format = CurrentFormat()
) -> ZarrLocation | None:
    """Convert a path string or URL to a ZarrLocation subclass.

    :param path: Path to parse.
    :param mode: Mode to open in.
    :param fmt: Version of the OME-NGFF spec to open path with.

    :return: `ZarrLocation`.
        If mode is 'r', and the path does not exist returns None.
        If there is an error opening the path, also returns None.

    >>> parse_url('does-not-exist')
    """
    try:
        loc = ZarrLocation(path, mode=mode, fmt=fmt)
        if "r" in mode and not loc.exists():
            return None
        else:
            return loc
    except Exception:
        LOGGER.exception("exception on parsing (stacktrace at DEBUG)")
        LOGGER.debug("stacktrace:", exc_info=True)
        return None
