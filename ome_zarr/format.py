"""Hierarchy of format OME-Zarr implementations."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional

from zarr.storage import FSStore

LOGGER = logging.getLogger("ome_zarr.format")


def format_from_version(version: str) -> "Format":

    for fmt in format_implementations():
        if fmt.version == version:
            return fmt
    raise ValueError(f"Version {version} not recognized")


def format_implementations() -> Iterator["Format"]:
    """
    Return an instance of each format implementation, newest to oldest.
    """
    yield FormatV04()
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
    def version(self) -> str:  # pragma: no cover
        raise NotImplementedError()

    @abstractmethod
    def matches(self, metadata: dict) -> bool:  # pragma: no cover
        raise NotImplementedError()

    @abstractmethod
    def init_store(self, path: str, mode: str = "r") -> FSStore:
        raise NotImplementedError()

    # @abstractmethod
    def init_channels(self) -> None:  # pragma: no cover
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

    @abstractmethod
    def generate_well_dict(
        self, well: str, rows: List[str], columns: List[str]
    ) -> dict:  # pragma: no cover
        raise NotImplementedError()

    @abstractmethod
    def validate_well_dict(
        self, well: dict, rows: List[str], columns: List[str]
    ) -> None:  # pragma: no cover
        raise NotImplementedError()

    @abstractmethod
    def validate_coordinate_transformations(
        self,
        shapes: List[tuple],
        coordinateTransformations: List[List[Dict[str, Any]]] = None,
    ) -> Optional[List[List[Dict[str, Any]]]]:  # pragma: no cover
        raise NotImplementedError()


class FormatV01(Format):
    """
    Initial format. (2020)
    """

    REQUIRED_PLATE_WELL_KEYS: Dict[str, type] = {"path": str}

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

    def generate_well_dict(
        self, well: str, rows: List[str], columns: List[str]
    ) -> dict:
        return {"path": str(well)}

    def validate_well_dict(
        self, well: dict, rows: List[str], columns: List[str]
    ) -> None:
        if any(e not in self.REQUIRED_PLATE_WELL_KEYS for e in well.keys()):
            LOGGER.debug("f{well} contains unspecified keys")
        for key, key_type in self.REQUIRED_PLATE_WELL_KEYS.items():
            if key not in well:
                raise ValueError(f"{well} must contain a {key} key of type {key_type}")
            if not isinstance(well[key], key_type):
                raise ValueError(f"{well} path must be of {key_type} type")

    def validate_coordinate_transformations(
        self,
        shapes: List[tuple],
        coordinateTransformations: List[List[Dict[str, Any]]] = None,
    ) -> Optional[List[List[Dict[str, Any]]]]:
        return None


class FormatV02(FormatV01):
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
            "normalize_keys": False,
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


class FormatV04(FormatV03):
    """
    Changelog: axes is list of dicts,
    introduce coordinateTransformations in multiscales (Nov 2021)
    """

    REQUIRED_PLATE_WELL_KEYS = {"path": str, "rowIndex": int, "columnIndex": int}

    @property
    def version(self) -> str:
        return "0.4"

    def generate_well_dict(
        self, well: str, rows: List[str], columns: List[str]
    ) -> dict:
        row, column = well.split("/")
        if row not in rows:
            raise ValueError(f"{row} is not defined in the list of rows")
        rowIndex = rows.index(row)
        if column not in columns:
            raise ValueError(f"{column} is not defined in the list of columns")
        columnIndex = columns.index(column)
        return {"path": str(well), "rowIndex": rowIndex, "columnIndex": columnIndex}

    def validate_well_dict(
        self, well: dict, rows: List[str], columns: List[str]
    ) -> None:
        super().validate_well_dict(well, rows, columns)
        if len(well["path"].split("/")) != 2:
            raise ValueError(f"{well} path must exactly be composed of 2 groups")
        row, column = well["path"].split("/")
        if row not in rows:
            raise ValueError(f"{row} is not defined in the plate rows")
        if well["rowIndex"] != rows.index(row):
            raise ValueError(f"Mismatching row index for {well}")
        if column not in columns:
            raise ValueError(f"{column} is not defined in the plate columns")
        if well["columnIndex"] != columns.index(column):
            raise ValueError(f"Mismatching column index for {well}")

    def validate_coordinate_transformations(
        self,
        shapes: List[tuple],
        coordinateTransformations: List[List[Dict[str, Any]]] = None,
    ) -> Optional[List[List[Dict[str, Any]]]]:
        """
        Validates that a list of dicts contains a 'scale' transformation

        Raises ValueError if no 'scale' found or doesn't match ndim
        """
        data_shape = shapes[0]
        trans: List[List[Dict[str, Any]]] = []
        if coordinateTransformations is None:
            # calculate minimal 'scale' transform based on pyramid dims
            for shape in shapes:
                assert len(shape) == len(data_shape)
                scale = [full / level for full, level in zip(data_shape, shape)]
                trans.append([{"type": "scale", "scale": scale}])
            coordinateTransformations = trans
        else:
            ct_count = len(coordinateTransformations)
            if ct_count != len(shapes):
                raise ValueError(
                    "coordinateTransformations count: %s must match datasets %s"
                    % (ct_count, len(shapes))
                )
            for shape, transform in zip(shapes, coordinateTransformations):
                self.validate_dataset_transformations(transform, len(shape))
        return coordinateTransformations

    def validate_dataset_transformations(
        self, transformations: List[Dict[str, Any]], ndim: int = None
    ) -> None:
        assert isinstance(transformations, list)
        scale_transfs = [trans for trans in transformations if trans["type"] == "scale"]
        if len(scale_transfs) == 0:
            raise ValueError("No scale items in coordinateTransforms")
        for trans in scale_transfs:
            scale = trans["scale"]
            if ndim is not None:
                if len(scale) != ndim:
                    raise ValueError(
                        "'scale' list %s must match number of image dimensions: %s"
                        % (scale, ndim)
                    )
            for value in scale:
                if not isinstance(value, (float, int)):
                    raise ValueError("'scale' values must all be numbers: %s" % scale)


CurrentFormat = FormatV04
