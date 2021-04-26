"""Hierarchy of format OME-Zarr implementations."""

from abc import ABC, abstractmethod

from zarr.storage import FSStore

from ome_zarr.io import ZarrLocation


def detect_version(loc: ZarrLocation) -> "Format":
    """
    """
    if "0.2":
        return FormatV2()
    else:
        return FormatV1()


class Format(ABC):
    @property
    @abstractmethod
    def version(self) -> str:
        raise NotImplementedError()

    # @abstractmethod
    def init_store(self) -> FSStore:
        raise NotImplementedError()

    # @abstractmethod
    def init_channels(self) -> None:
        raise NotImplementedError()


class FormatV1(Format):
    """
    Initial format
    """

    @property
    def version(self) -> str:
        return "0.1"


class FormatV2(Format):
    """
    Changelog: move to nested storage
    """

    @property
    def version(self) -> str:
        return "0.2"
