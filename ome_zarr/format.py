"""Hierarchy of format OME-Zarr implementations."""

import logging
from abc import ABC, abstractmethod
from typing import Iterator, Optional

from zarr.storage import FSStore

LOGGER = logging.getLogger("ome_zarr.format")


def format_implementations() -> Iterator["Format"]:
    """
    Return an instance of each format implementation, newest to oldest.
    """
    yield FormatV03()
    yield FormatV02()
    yield FormatV01()


def detect_format(metadata: dict) -> "Format":
    """
    Give each format implementation a chance to take ownership of the
    given metadata. If none matches, a CurrentFormat is returned.
    """

    if metadata:
        for fmt in format_implementations():
            if fmt.matches(metadata):
                return fmt

    return CurrentFormat()


class Format(ABC):
    @property
    @abstractmethod
    def version(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def matches(self, metadata: dict) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def init_store(self, path: str, mode: str = "r") -> FSStore:
        raise NotImplementedError()

    # @abstractmethod
    def init_channels(self) -> None:
        raise NotImplementedError()

    def _get_multiscale_version(self, metadata: dict) -> Optional[str]:
        multiscales = metadata.get("multiscales", [])
        if multiscales:
            dataset = multiscales[0]
            return dataset.get("version", None)
        return None

    def __repr__(self) -> str:
        return self.__class__.__name__

    def __eq__(self, other: object) -> bool:
        return self.__class__ == other.__class__


class FormatV01(Format):
    """
    Initial format. (2020)
    """

    @property
    def version(self) -> str:
        return "0.1"

    def matches(self, metadata: dict) -> bool:
        version = self._get_multiscale_version(metadata)
        LOGGER.debug(f"V01:{version} v. {self.version}")
        return version == self.version

    def init_store(self, path: str, mode: str = "r") -> FSStore:
        store = FSStore(path, mode=mode, dimension_separator=".")
        LOGGER.debug(f"Created legacy flat FSStore({path}, {mode})")
        return store


class FormatV02(Format):
    """
    Changelog: move to nested storage (April 2021)
    """

    @property
    def version(self) -> str:
        return "0.2"

    def matches(self, metadata: dict) -> bool:
        version = self._get_multiscale_version(metadata)
        LOGGER.debug(f"{self.version} matches {version}?")
        return version == self.version

    def init_store(self, path: str, mode: str = "r") -> FSStore:
        """
        Not ideal. Stores should remain hidden
        TODO: could also check dimension_separator
        """

        kwargs = {
            "dimension_separator": "/",
            "normalize_keys": True,
        }

        mkdir = True
        if "r" in mode or path.startswith("http"):
            # Could be simplified on the fsspec side
            mkdir = False
        if mkdir:
            kwargs["auto_mkdir"] = True

        store = FSStore(
            path,
            mode=mode,
            **kwargs,
        )  # TODO: open issue for using Path
        LOGGER.debug(f"Created nested FSStore({path}, {mode}, {kwargs})")
        return store


class FormatV03(FormatV02):  # inherits from V02 to avoid code duplication
    """
    Changelog: variable number of dimensions (up to 5),
    introduce axes field in multiscales (June 2021)
    """

    @property
    def version(self) -> str:
        return "0.3"


CurrentFormat = FormatV03
