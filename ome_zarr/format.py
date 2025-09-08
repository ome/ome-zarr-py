"""Hierarchy of format OME-Zarr implementations."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from typing import Any

from zarr.storage import FsspecStore, LocalStore

LOGGER = logging.getLogger("ome_zarr.format")


def format_from_version(version: str) -> "Format":
    for fmt in format_implementations():
        # Support floating-point versions like `0.2`
        if isinstance(version, float):
            version = str(version)

        if fmt.version == version:
            return fmt
    raise ValueError(f"Version {version} not recognized")


def format_implementations() -> Iterator["Format"]:
    """
    Return an instance of each format implementation, newest to oldest.
    """
    yield FormatV05()
    yield FormatV04()
    yield FormatV03()
    yield FormatV02()
    yield FormatV01()


def detect_format(metadata: dict, default: "Format") -> "Format":
    """
    Give each format implementation a chance to take ownership of the
    given metadata. If none matches, the default value will be returned.
    """

    if metadata:
        for fmt in format_implementations():
            if fmt.matches(metadata):
                return fmt

    return default


class Format(ABC):
    """
    Abstract base class for format implementations.
    """

    @property
    @abstractmethod
    def version(self) -> str:  # pragma: no cover
        raise NotImplementedError()

    @property
    @abstractmethod
    def zarr_format(self) -> int:  # pragma: no cover
        raise NotImplementedError()

    @property
    @abstractmethod
    def chunk_key_encoding(self) -> dict[str, str]:  # pragma: no cover
        raise NotImplementedError()

    @abstractmethod
    def matches(self, metadata: dict) -> bool:  # pragma: no cover
        raise NotImplementedError()

    @abstractmethod
    def init_store(self, path: str, mode: str = "r") -> FsspecStore | LocalStore:
        raise NotImplementedError()

    # @abstractmethod
    def init_channels(self) -> None:  # pragma: no cover
        raise NotImplementedError()

    def _get_metadata_version(self, metadata: dict) -> str | None:
        """
        Checks the metadata dict for a version

        Returns the version of the first object found in the metadata,
        checking for 'multiscales', 'plate', 'well' etc
        """
        multiscales = metadata.get("multiscales", [])
        if multiscales:
            dataset = multiscales[0]
            return dataset.get("version", None)
        for name in ["plate", "well", "image-label"]:
            obj = metadata.get(name)
            if obj:
                return obj.get("version", None)
        return None

    def __repr__(self) -> str:
        return self.__class__.__name__

    def __eq__(self, other: object) -> bool:
        return self.__class__ == other.__class__

    @abstractmethod
    def generate_well_dict(
        self, well: str, rows: list[str], columns: list[str]
    ) -> dict:  # pragma: no cover
        raise NotImplementedError()

    @abstractmethod
    def validate_well_dict(
        self, well: dict, rows: list[str], columns: list[str]
    ) -> None:  # pragma: no cover
        raise NotImplementedError()

    @abstractmethod
    def generate_coordinate_transformations(
        self, shapes: list[tuple]
    ) -> list[list[dict[str, Any]]] | None:  # pragma: no cover
        raise NotImplementedError()

    @abstractmethod
    def validate_coordinate_transformations(
        self,
        ndim: int,
        nlevels: int,
        coordinate_transformations: list[list[dict[str, Any]]] | None = None,
    ) -> list[list[dict[str, Any]]] | None:  # pragma: no cover
        raise NotImplementedError()


class FormatV01(Format):
    """
    Initial format. (2020)
    """

    REQUIRED_PLATE_WELL_KEYS: Mapping[str, type] = {"path": str}

    @property
    def version(self) -> str:
        return "0.1"

    @property
    def zarr_format(self) -> int:
        return 2

    @property
    def chunk_key_encoding(self) -> dict[str, str]:
        return {"name": "v2", "separator": "."}

    def matches(self, metadata: dict) -> bool:
        version = self._get_metadata_version(metadata)
        LOGGER.debug("%s matches %s?", self.version, version)
        return version == self.version

    def init_store(self, path: str, mode: str = "r") -> FsspecStore | LocalStore:
        """
        Not ideal. Stores should remain hidden
        "dimension_separator" is specified at array creation time
        """

        read_only = mode == "r"
        if path.startswith(("http", "s3")):
            store = FsspecStore.from_url(
                path,
                storage_options=None,
                read_only=read_only,
            )
        else:
            # No other kwargs supported
            store = LocalStore(path, read_only=read_only)
        LOGGER.debug("Created nested FsspecStore(%s, %s)", path, mode)
        return store

    def generate_well_dict(
        self, well: str, rows: list[str], columns: list[str]
    ) -> dict:
        return {"path": str(well)}

    def validate_well_dict(
        self, well: dict, rows: list[str], columns: list[str]
    ) -> None:
        if any(e not in self.REQUIRED_PLATE_WELL_KEYS for e in well):
            LOGGER.debug("%s contains unspecified keys", well)
        for key, key_type in self.REQUIRED_PLATE_WELL_KEYS.items():
            if key not in well:
                raise ValueError(
                    "%s must contain a %s key of type %s", well, key, key_type
                )
            if not isinstance(well[key], key_type):
                raise ValueError("%s path must be of %s type", well, key_type)

    def generate_coordinate_transformations(
        self, shapes: list[tuple]
    ) -> list[list[dict[str, Any]]] | None:
        return None

    def validate_coordinate_transformations(
        self,
        ndim: int,
        nlevels: int,
        coordinate_transformations: list[list[dict[str, Any]]] | None = None,
    ) -> None:
        return None


class FormatV02(FormatV01):
    """
    Changelog: move to nested storage (April 2021)
    """

    @property
    def version(self) -> str:
        return "0.2"

    @property
    def chunk_key_encoding(self) -> dict[str, str]:
        return {"name": "v2", "separator": "/"}


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
    introduce coordinate_transformations in multiscales (Nov 2021)
    """

    REQUIRED_PLATE_WELL_KEYS: Mapping[str, type] = {
        "path": str,
        "rowIndex": int,
        "columnIndex": int,
    }

    @property
    def version(self) -> str:
        return "0.4"

    def generate_well_dict(
        self, well: str, rows: list[str], columns: list[str]
    ) -> dict:
        row, column = well.split("/")
        if row not in rows:
            raise ValueError("%s is not defined in the list of rows", row)
        rowIndex = rows.index(row)
        if column not in columns:
            raise ValueError("%s is not defined in the list of columns", column)
        columnIndex = columns.index(column)
        return {"path": str(well), "rowIndex": rowIndex, "columnIndex": columnIndex}

    def validate_well_dict(
        self, well: dict, rows: list[str], columns: list[str]
    ) -> None:
        super().validate_well_dict(well, rows, columns)
        if len(well["path"].split("/")) != 2:
            raise ValueError("%s path must exactly be composed of 2 groups", well)
        row, column = well["path"].split("/")
        if row not in rows:
            raise ValueError("%s is not defined in the plate rows", row)
        if well["rowIndex"] != rows.index(row):
            raise ValueError("Mismatching row index for %s", well)
        if column not in columns:
            raise ValueError("%s is not defined in the plate columns", column)
        if well["columnIndex"] != columns.index(column):
            raise ValueError("Mismatching column index for %s", well)

    def generate_coordinate_transformations(
        self, shapes: list[tuple]
    ) -> list[list[dict[str, Any]]] | None:
        data_shape = shapes[0]
        coordinate_transformations: list[list[dict[str, Any]]] = []
        # calculate minimal 'scale' transform based on pyramid dims
        for shape in shapes:
            assert len(shape) == len(data_shape)
            scale = [full / level for full, level in zip(data_shape, shape)]
            coordinate_transformations.append([{"type": "scale", "scale": scale}])

        return coordinate_transformations

    def validate_coordinate_transformations(
        self,
        ndim: int,
        nlevels: int,
        coordinate_transformations: list[list[dict[str, Any]]] | None = None,
    ) -> None:
        """
        Validates that a list of dicts contains a 'scale' transformation

        Raises ValueError if no 'scale' found or doesn't match ndim.

        :param ndim: Number of image dimensions.
        """

        if coordinate_transformations is None:
            raise ValueError("coordinate_transformations must be provided")
        ct_count = len(coordinate_transformations)
        if ct_count != nlevels:
            raise ValueError(
                f"coordinate_transformations count: {ct_count} must match "
                f"datasets {nlevels}"
            )
        for transformations in coordinate_transformations:
            assert isinstance(transformations, list)
            types = [t.get("type", None) for t in transformations]
            if any(t is None for t in types):
                raise ValueError(f"Missing type in: {transformations}")
            # validate scales...
            if sum(t == "scale" for t in types) != 1:
                raise ValueError(
                    "Must supply 1 'scale' item in coordinate_transformations"
                )
            # first transformation must be scale
            if types[0] != "scale":
                raise ValueError("First coordinate_transformations must be 'scale'")
            first = transformations[0]
            if "scale" not in transformations[0]:
                raise ValueError(f"Missing scale argument in: {first}")
            scale = first["scale"]
            if len(scale) != ndim:
                raise ValueError(
                    f"'scale' list {scale} must match "
                    f"number of image dimensions: {ndim}"
                )
            for value in scale:
                if not isinstance(value, (float, int)):
                    raise ValueError(f"'scale' values must all be numbers: {scale}")

            # validate translations...
            translation_types = [t == "translation" for t in types]
            if sum(translation_types) > 1:
                raise ValueError(
                    "Must supply 0 or 1 'translation' item in"
                    "coordinate_transformations"
                )
            elif sum(translation_types) == 1:
                transformation = transformations[types.index("translation")]
                if "translation" not in transformation:
                    raise ValueError(f"Missing scale argument in: {first}")
                translation = transformation["translation"]
                if len(translation) != ndim:
                    raise ValueError(
                        f"'translation' list {translation} must match "
                        f"image dimensions count: {ndim}"
                    )
                for value in translation:
                    if not isinstance(value, (float, int)):
                        raise ValueError(
                            f"'translation' values must all be numbers: {translation}"
                        )


class FormatV05(FormatV04):
    """
    Changelog: added FormatV05 (May 2025): writing not supported yet
    """

    @property
    def version(self) -> str:
        return "0.5"

    @property
    def zarr_format(self) -> int:
        return 3

    @property
    def chunk_key_encoding(self) -> dict[str, str]:
        # this is default for Zarr v3. Could return None?
        return {"name": "default", "separator": "/"}


CurrentFormat = FormatV05
