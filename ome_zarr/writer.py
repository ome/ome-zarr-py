"""Image writer utility

"""
import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import zarr

from .axes import Axes
from .format import CurrentFormat, Format
from .scale import Scaler
from .types import JSONDict

LOGGER = logging.getLogger("ome_zarr.writer")


def _get_valid_axes(
    ndim: int = None,
    axes: Union[str, List[str], List[Dict[str, str]]] = None,
    fmt: Format = CurrentFormat(),
) -> Union[None, List[str], List[Dict[str, str]]]:
    """Returns list of axes valid for fmt.version or raise exception if invalid"""

    if fmt.version in ("0.1", "0.2"):
        if axes is not None:
            LOGGER.info("axes ignored for version 0.1 or 0.2")
        return None

    # We can guess axes for 2D and 5D data
    if axes is None:
        if ndim == 2:
            axes = ["y", "x"]
            LOGGER.info("Auto using axes %s for 2D data" % axes)
        elif ndim == 5:
            axes = ["t", "c", "z", "y", "x"]
            LOGGER.info("Auto using axes %s for 5D data" % axes)
        else:
            raise ValueError(
                "axes must be provided. Can't be guessed for 3D or 4D data"
            )

    # axes may be string e.g. "tczyx"
    if isinstance(axes, str):
        axes = list(axes)

    if ndim is not None and len(axes) != ndim:
        raise ValueError(
            f"axes length ({len(axes)}) must match number of dimensions ({ndim})"
        )

    # valiates on init
    axes_obj = Axes(axes, fmt)

    return axes_obj.to_list(fmt)


def _validate_well_images(
    images: List[Union[str, dict]], fmt: Format = CurrentFormat()
) -> List[dict]:

    VALID_KEYS = [
        "acquisition",
        "path",
    ]
    validated_images = []
    for image in images:
        if isinstance(image, str):
            validated_images.append({"path": str(image)})
        elif isinstance(image, dict):
            if any(e not in VALID_KEYS for e in image.keys()):
                LOGGER.debug("f{image} contains unspecified keys")
            if "path" not in image:
                raise ValueError(f"{image} must contain a path key")
            if not isinstance(image["path"], str):
                raise ValueError(f"{image} path must be of string type")
            if "acquisition" in image and not isinstance(image["acquisition"], int):
                raise ValueError(f"{image} acquisition must be of int type")
            validated_images.append(image)
        else:
            raise ValueError(f"Unrecognized type for {image}")
    return validated_images


def _validate_plate_acquisitions(
    acquisitions: List[Dict], fmt: Format = CurrentFormat()
) -> List[Dict]:

    VALID_KEYS = [
        "id",
        "name",
        "maximumfieldcount",
        "description",
        "starttime",
        "endtime",
    ]

    for acquisition in acquisitions:
        if not isinstance(acquisition, dict):
            raise ValueError(f"{acquisition} must be a dictionary")
        if any(e not in VALID_KEYS for e in acquisition.keys()):
            LOGGER.debug("f{acquisition} contains unspecified keys")
        if "id" not in acquisition:
            raise ValueError(f"{acquisition} must contain an id key")
        if not isinstance(acquisition["id"], int):
            raise ValueError(f"{acquisition} id must be of int type")
    return acquisitions


def _validate_plate_rows_columns(
    rows_or_columns: List[str],
    fmt: Format = CurrentFormat(),
) -> List[dict]:

    if len(set(rows_or_columns)) != len(rows_or_columns):
        raise ValueError(f"{rows_or_columns} must contain unique elements")
    validated_list = []
    for element in rows_or_columns:
        if not element.isalnum():
            raise ValueError(f"{element} must contain alphanumeric characters")
        validated_list.append({"name": str(element)})
    return validated_list


def _validate_datasets(
    datasets: List[dict], dims: int, fmt: Format = CurrentFormat()
) -> List[Dict]:

    if datasets is None or len(datasets) == 0:
        raise ValueError("Empty datasets list")
    transformations = []
    for dataset in datasets:
        if isinstance(dataset, dict):
            if not dataset.get("path"):
                raise ValueError("no 'path' in dataset")
            transformation = dataset.get("coordinateTransformations")
            # transformation may be None for < 0.4 - validated below
            if transformation is not None:
                transformations.append(transformation)
        else:
            raise ValueError(f"Unrecognized type for {dataset}")

    fmt.validate_coordinate_transformations(dims, len(datasets), transformations)
    return datasets


def _validate_plate_wells(
    wells: List[Union[str, dict]],
    rows: List[str],
    columns: List[str],
    fmt: Format = CurrentFormat(),
) -> List[dict]:

    validated_wells = []
    if wells is None or len(wells) == 0:
        raise ValueError("Empty wells list")
    for well in wells:
        if isinstance(well, str):
            well_dict = fmt.generate_well_dict(well, rows, columns)
            fmt.validate_well_dict(well_dict, rows, columns)
            validated_wells.append(well_dict)
        elif isinstance(well, dict):
            fmt.validate_well_dict(well, rows, columns)
            validated_wells.append(well)
        else:
            raise ValueError(f"Unrecognized type for {well}")
    return validated_wells


def write_multiscale(
    pyramid: List,
    group: zarr.Group,
    chunks: Union[Tuple[Any, ...], int] = None,
    fmt: Format = CurrentFormat(),
    axes: Union[str, List[str], List[Dict[str, str]]] = None,
    coordinate_transformations: List[List[Dict[str, Any]]] = None,
    storage_options: Union[JSONDict, List[JSONDict]] = None,
) -> None:
    """
    Write a pyramid with multiscale metadata to disk.

    pyramid: List of np.ndarray
      the image data to save. Largest level first
      All image arrays MUST be up to 5-dimensional with dimensions
      ordered (t, c, z, y, x)
    group: zarr.Group
      the group within the zarr store to store the data in
    chunks: int or tuple of ints,
      size of the saved chunks to store the image
    fmt: Format
      The format of the ome_zarr data which should be used.
      Defaults to the most current.
    axes: str or list of str or list of dict
      List of axes dicts, or names. Not needed for v0.1 or v0.2
      or if 2D. Otherwise this must be provided
    coordinate_transformations: 2Dlist of dict
      For each path, we have a List of transformation Dicts.
      Each list of dicts are added to each datasets in order
      and must include a 'scale' transform.
    storage_options: dict or list of dict
      Options to be passed on to the storage backend. A list would need to match
      the number of datasets in a multiresolution pyramid. One can provide
      different chunk size for each level of a pyramind using this option.
    """

    dims = len(pyramid[0].shape)
    axes = _get_valid_axes(dims, axes, fmt)

    datasets: List[dict] = []
    for path, data in enumerate(pyramid):
        options = {}
        if storage_options:
            options = (
                storage_options
                if not isinstance(storage_options, list)
                else storage_options[path]
            )
        if "chunks" not in options:
            options["chunks"] = chunks
        group.create_dataset(str(path), data=data, **options)
        datasets.append({"path": str(path)})

    if coordinate_transformations is None:
        shapes = [data.shape for data in pyramid]
        coordinate_transformations = fmt.generate_coordinate_transformations(shapes)

    # we validate again later, but this catches length mismatch before zip(datasets...)
    fmt.validate_coordinate_transformations(
        dims, len(pyramid), coordinate_transformations
    )
    if coordinate_transformations is not None:
        for dataset, transform in zip(datasets, coordinate_transformations):
            dataset["coordinateTransformations"] = transform

    write_multiscales_metadata(group, datasets, fmt, axes)


def write_multiscales_metadata(
    group: zarr.Group,
    datasets: List[dict],
    fmt: Format = CurrentFormat(),
    axes: Union[str, List[str], List[Dict[str, str]]] = None,
) -> None:
    """
    Write the multiscales metadata in the group.

    group: zarr.Group
      the group within the zarr store to write the metadata in.
    datasets: list of dicts
      The list of datasets (dicts) for this multiscale image.
      Each dict must include 'path' and a 'coordinateTransformations'
      list for version 0.4 or later that must include a 'scale' transform.
    fmt: Format
      The format of the ome_zarr data which should be used.
      Defaults to the most current.
    axes: list of str or list of dicts
      the names of the axes. e.g. ["t", "c", "z", "y", "x"].
      Ignored for versions 0.1 and 0.2. Required for version 0.3 or greater.
    """

    ndim = -1
    if axes is not None:
        if fmt.version in ("0.1", "0.2"):
            LOGGER.info("axes ignored for version 0.1 or 0.2")
            axes = None
        else:
            axes = _get_valid_axes(axes=axes, fmt=fmt)
            if axes is not None:
                ndim = len(axes)

    multiscales = [
        {
            "version": fmt.version,
            "datasets": _validate_datasets(datasets, ndim, fmt),
        }
    ]
    if axes is not None:
        multiscales[0]["axes"] = axes

    group.attrs["multiscales"] = multiscales


def write_plate_metadata(
    group: zarr.Group,
    rows: List[str],
    columns: List[str],
    wells: List[Union[str, dict]],
    fmt: Format = CurrentFormat(),
    acquisitions: List[dict] = None,
    field_count: int = None,
    name: str = None,
) -> None:
    """
    Write the plate metadata in the group.

    group: zarr.Group
      the group within the zarr store to write the metadata in.
    rows: list of str
      The list of names for the plate rows
    columns: list of str
      The list of names for the plate columns
    wells: list of str or dict
      The list of paths for the well groups
    fmt: Format
      The format of the ome_zarr data which should be used.
      Defaults to the most current.
    name: str
      The plate name
    field_count: int
      The maximum number of fields per view across wells
    acquisitions: list of dict
      A list of the various plate acquisitions
    """

    plate: Dict[str, Union[str, int, List[Dict]]] = {
        "columns": _validate_plate_rows_columns(columns),
        "rows": _validate_plate_rows_columns(rows),
        "wells": _validate_plate_wells(wells, rows, columns, fmt=fmt),
        "version": fmt.version,
    }
    if name is not None:
        plate["name"] = name
    if field_count is not None:
        plate["field_count"] = field_count
    if acquisitions is not None:
        plate["acquisitions"] = _validate_plate_acquisitions(acquisitions)
    group.attrs["plate"] = plate


def write_well_metadata(
    group: zarr.Group,
    images: List[Union[str, dict]],
    fmt: Format = CurrentFormat(),
) -> None:
    """
    Write the well metadata in the group.

    group: zarr.Group
      the group within the zarr store to write the metadata in.
    image_paths: list of str or dict
      The list of paths for the well images
    image_acquisitions: list of int
      The list of acquisitions for the well images
    fmt: Format
      The format of the ome_zarr data which should be used.
      Defaults to the most current.
    """

    well = {
        "images": _validate_well_images(images),
        "version": fmt.version,
    }
    group.attrs["well"] = well


def write_image(
    image: np.ndarray,
    group: zarr.Group,
    chunks: Union[Tuple[Any, ...], int] = None,
    byte_order: Union[str, List[str]] = "tczyx",
    scaler: Scaler = Scaler(),
    fmt: Format = CurrentFormat(),
    axes: Union[str, List[str], List[Dict[str, str]]] = None,
    coordinate_transformations: List[List[Dict[str, Any]]] = None,
    storage_options: Union[JSONDict, List[JSONDict]] = None,
    **metadata: JSONDict,
) -> None:
    """Writes an image to the zarr store according to ome-zarr specification

    image: np.ndarray
      the image data to save. A downsampling of the data will be computed
      if the scaler argument is non-None.
      Image array MUST be up to 5-dimensional with dimensions
      ordered (t, c, z, y, x)
    group: zarr.Group
      the group within the zarr store to store the data in
    chunks: int or tuple of ints,
      size of the saved chunks to store the image
    byte_order: str or list of str, default "tczyx"
      combination of the letters defining the order
      in which the dimensions are saved
    scaler: Scaler
      Scaler implementation for downsampling the image argument. If None,
      no downsampling will be performed.
    fmt: Format
      The format of the ome_zarr data which should be used.
      Defaults to the most current.
    axes: str or list of str or list of dict
      List of axes dicts, or names. Not needed for v0.1 or v0.2
      or if 2D. Otherwise this must be provided
    coordinate_transformations: 2Dlist of dict
      For each resolution, we have a List of transformation Dicts (not validated).
      Each list of dicts are added to each datasets in order.
    storage_options: dict or list of dict
      Options to be passed on to the storage backend. A list would need to match
      the number of datasets in a multiresolution pyramid. One can provide
      different chunk size for each level of a pyramind using this option.
    """

    if image.ndim > 5:
        raise ValueError("Only images of 5D or less are supported")

    if fmt.version in ("0.1", "0.2"):
        # v0.1 and v0.2 are strictly 5D
        shape_5d: Tuple[Any, ...] = (*(1,) * (5 - image.ndim), *image.shape)
        image = image.reshape(shape_5d)
        # and we don't need axes
        axes = None

    # check axes before trying to scale
    _get_valid_axes(image.ndim, axes, fmt)

    if chunks is not None:
        chunks = _retuple(chunks, image.shape)

    if scaler is not None:
        if image.shape[-1] == 1 or image.shape[-2] == 1:
            raise ValueError(
                "Can't downsample if size of x or y dimension is 1. "
                "Shape: %s" % (image.shape,)
            )
        image = scaler.nearest(image)
    else:
        LOGGER.debug("disabling pyramid")
        image = [image]

    write_multiscale(
        image,
        group,
        chunks=chunks,
        fmt=fmt,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
        storage_options=storage_options,
    )
    group.attrs.update(metadata)


def _retuple(
    chunks: Union[Tuple[Any, ...], int], shape: Tuple[Any, ...]
) -> Tuple[Any, ...]:

    _chunks: Tuple[Any, ...]
    if isinstance(chunks, int):
        _chunks = (chunks,)
    else:
        _chunks = chunks

    dims_to_add = len(shape) - len(_chunks)

    return (*shape[:dims_to_add], *_chunks)
