"""Image writer utility"""

import logging
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import dask.array as da
import numpy as np
import zarr
from numcodecs import Blosc

from . import USE_DASK_ARRAY_KWARGS
from .axes import Axes
from .format import CurrentFormat, Format, FormatV01, FormatV02, FormatV03, FormatV04
from .scale import Methods, Scaler
from .types import JSONDict

LOGGER = logging.getLogger("ome_zarr.writer")

ListOfArrayLike = list[da.Array] | list[np.ndarray]
ArrayLike: TypeAlias = da.Array | np.ndarray

AxesType = str | list[str] | list[dict[str, str]] | None

SPATIAL_DIMS = ("x", "y", "z")


def _get_valid_axes(
    ndim: int | None = None,
    axes: AxesType = None,
    axes_units: dict[str, str] | None = None,
    fmt: Format = CurrentFormat(),
) -> list[str] | list[dict[str, str]] | None:
    """Returns list of axes valid for fmt.version or raise exception if invalid"""

    if fmt.version in ("0.1", "0.2"):
        if axes is not None:
            LOGGER.info("axes ignored for version 0.1 or 0.2")
        return None

    # We can guess axes for 2D and 5D data
    if axes is None:
        if ndim == 2:
            axes = ["y", "x"]
            LOGGER.info("Auto using axes %s for 2D data", axes)
        elif ndim == 5:
            axes = ["t", "c", "z", "y", "x"]
            LOGGER.info("Auto using axes %s for 5D data", axes)
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

    # validates on init
    axes_obj = Axes(axes, axes_units, fmt)

    return axes_obj.to_list(fmt)


def _extract_dims_from_axes(
    axes: list[str] | list[dict[str, str]] | None,
) -> Sequence[str]:
    """Extract dimension names from axes, with proper type narrowing.

    Parameters
    ----------
    axes : list[str] | list[dict[str, str]] | None
        Axes returned from _get_valid_axes (must not be None).

    Returns
    -------
    Sequence[str]
        Dimension names as tuple.

    Raises
    ------
    ValueError
        If axes is None.
    """
    if axes is None:
        # only the case for v0.1 and v0.2, which are always 5D
        return ("t", "c", "z", "y", "x")

    # axes is expected to be a list of strings or a list of dicts with 'name'
    if all(isinstance(s, str) for s in axes):
        return tuple(str(s) for s in axes)

    if all(isinstance(s, dict) and "name" in s for s in axes):
        names: list[str] = []
        for s in axes:
            # narrow type for mypy
            if not isinstance(s, dict) or "name" not in s:
                raise TypeError("`axes` must be a list of dicts containing 'name'")
            names.append(str(s["name"]))
        return tuple(names)

    raise TypeError(
        "`axes` must be a list of strings or a list of dicts containing 'name'"
    )


def _validate_well_images(
    images: list[str | dict], fmt: Format = CurrentFormat()
) -> list[dict]:
    VALID_KEYS = [
        "acquisition",
        "path",
    ]
    validated_images = []
    for image in images:
        if isinstance(image, str):
            validated_images.append({"path": str(image)})
        elif isinstance(image, dict):
            if any(e not in VALID_KEYS for e in image):
                LOGGER.debug("%s contains unspecified keys", image)
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
    acquisitions: list[dict], fmt: Format = CurrentFormat()
) -> list[dict]:
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
        if any(e not in VALID_KEYS for e in acquisition):
            LOGGER.debug("%s contains unspecified keys", acquisition)
        if "id" not in acquisition:
            raise ValueError(f"{acquisition} must contain an id key")
        if not isinstance(acquisition["id"], int):
            raise ValueError(f"{acquisition} id must be of int type")
    return acquisitions


def _validate_plate_rows_columns(
    rows_or_columns: list[str],
    fmt: Format = CurrentFormat(),
) -> list[dict]:
    if len(set(rows_or_columns)) != len(rows_or_columns):
        raise ValueError(f"{rows_or_columns} must contain unique elements")
    validated_list = []
    for element in rows_or_columns:
        if not element.isalnum():
            raise ValueError(f"{element} must contain alphanumeric characters")
        validated_list.append({"name": str(element)})
    return validated_list


def _validate_datasets(
    datasets: list[dict], dims: int, fmt: Format = CurrentFormat()
) -> list[dict]:
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
    wells: list[str | dict],
    rows: list[str],
    columns: list[str],
    fmt: Format = CurrentFormat(),
) -> list[dict]:
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


def _blosc_compressor() -> Blosc:
    """Return a Blosc compressor with zstd compression"""
    return Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)


def check_group_fmt(
    group: zarr.Group | str,
    fmt: Format | None = None,
    mode: str = "a",
) -> tuple[zarr.Group, Format]:
    """
    Create group if string, according to fmt
    OR check fmt is compatible with group
    """
    if isinstance(group, str):
        if not fmt:
            group = zarr.open_group(group, mode=mode)
        else:
            group = zarr.open_group(group, mode=mode, zarr_format=fmt.zarr_format)

    fmt = check_format(group, fmt)
    return group, fmt


def check_format(
    group: zarr.Group,
    fmt: Format | None = None,
) -> Format:
    """Check if the format is valid for the given group"""

    zarr_format = group.info._zarr_format
    if fmt is not None:
        if fmt.zarr_format != zarr_format:
            raise ValueError(
                f"Group is zarr_format: {zarr_format} but OME-Zarr {fmt.version} is {fmt.zarr_format}"
            )
    elif zarr_format == 2:
        fmt = FormatV04()
    elif zarr_format == 3:
        fmt = CurrentFormat()
    assert fmt is not None
    return fmt


def write_multiscale(
    pyramid: ListOfArrayLike,
    group: zarr.Group,
    fmt: Format | None = None,
    axes: AxesType = None,
    coordinate_transformations: list[list[dict[str, Any]]] | None = None,
    storage_options: JSONDict | list[JSONDict] | None = None,
    name: str | None = None,
    compute: bool = True,
    scale: dict[str, float] | None = None,
    axes_units: dict[str, str] | None = None,
) -> list:
    """
    Write a pyramid with precomputed multiscale resolution levels to disk.

    Parameters
    ----------
    pyramid: list of :class:`numpy.ndarray` or :class:`dask.array.Array`
        The image data to save. Largest level first. All image arrays MUST be up to
        5-dimensional with dimensions ordered (t, c, z, y, x)
    group: :class:`zarr.Group`
        The group within the zarr store to store the data in
    fmt: :class:`ome_zarr.format.Format`, optional
        The format of the ome_zarr data which should be used.
        Defaults to the most current (:class:`ome_zarr.format.CurrentFormat`).
    axes: list of str or list of dict, optional
        List of axes dicts, or names, i.e. ["t", "c", "z", "y", "x"].
    coordinate_transformations: list of list of dict, optional
        [DEPRECATED] For each resolution, a list of transformation dicts (not validated).
    storage_options: dict or list of dict, optional
        Options to be passed on to the storage backend.
        A list would need to match the number of datasets in a multiresolution pyramid.
        One can provide different chunk size and / or shards for each level of a pyramid using this
        option. Regarding the key, value pairs in the dictionar(y)(ies), these depend both on the zarr_format used
        for writing and the dask version being used. For dask version <=2025.11.0, please refer to
        https://zarr.readthedocs.io/en/stable/api/zarr/create/#zarr.create for arguments that can be passed on.
        For >=2026.3.0 and up, please refer to https://zarr.readthedocs.io/en/stable/api/zarr/create/#zarr.create_array.
        It might be that you have to adjust the version of the docs. Note that the docs will also mention the
        differences of allowed arguments between zarr_format 2 and 3.

        Note: for chunks the default of `auto` is not allowed. This because the argument here refers to zarr chunks and
        autochunking here can result in different chunks then for the dask array. This can cause inconsistent overlap
        between dask and zarr chunks, potentially resulting in corrupted data. The default will be that if no sharding
        is specified, that the chunks correspond to the dask chunksize. This is also the case when chunks are provided as
        `None` and no sharding is provided.
    name: str, optional
        The name of the image, to be included in the metadata. Defaults to "image".
    compute: bool, optional
        If true, compute immediately otherwise a list of :class:`dask.delayed.Delayed`
        is returned.
    scale: dict of str to float, optional
        The physical pixel size for each dimension, e.g. {"z": 0.1, "y": 0.1, "x": 0.5}.
        For each additional resolution level, the pixel sizes are derived from this
        base `scale` and the relative shapes of the arrays provided in `pyramid`.
        If not provided, defaults to 1.0 for all dimensions.
    axes_units: dict of str to str, optional
        The physical units for each dimension,
        e.g. {"t": "millisecond", "z": "micrometer", "y": "micrometer", "x": "micrometer"}.
    """
    from ome_zarr import OMEZarrImage, OMEZarrMultiscale

    group, fmt = check_group_fmt(group, fmt)
    dims = len(pyramid[0].shape)
    axes = _get_valid_axes(dims, axes, axes_units=axes_units, fmt=fmt)

    if scale is None:
        scale = dict.fromkeys(_extract_dims_from_axes(axes), 1.0)

    if name is None:
        name = "image"

    if coordinate_transformations is not None:
        msg = (
            "The 'coordinate_transformations' argument is deprecated and will "
            "be removed in a future version. Please use the `scale` argument "
            "to specify the physical pixel size for each dimension instead."
        )
        warnings.warn(msg, DeprecationWarning)

    images = []
    for level in pyramid:
        relative_factor = np.asarray(level.shape) / np.asarray(pyramid[0].shape)
        level_scale = {
            d: s * relative_factor[i] for i, (d, s) in enumerate(scale.items())
        }
        img = OMEZarrImage(
            data=level,
            axes=list(scale.keys()),
            scale=level_scale,
            axes_units=axes_units,
            name=name,
        )
        images.append(img)

    ms = OMEZarrMultiscale(
        image=images,
    )

    if fmt.version not in ("0.4", "0.5", "0.6.dev4"):
        raise ValueError(f"Unsupported format version: {fmt.version}")

    dask_delayed = ms.to_ome_zarr(
        group,
        version=cast(Literal["0.6.dev4", "0.5", "0.4"], fmt.version),
        compute=compute,
        storage_options=storage_options,
        overwrite=True,
    )

    return dask_delayed


def write_plate_metadata(
    group: zarr.Group | str,
    rows: list[str],
    columns: list[str],
    wells: list[str | dict],
    fmt: Format | None = None,
    acquisitions: list[dict] | None = None,
    field_count: int | None = None,
    name: str | None = None,
) -> None:
    """
    Write the plate metadata in the group.

    :type group: :class:`zarr.Group`
    :param group: The group or path to write the metadata in.
    :type rows: list of str
    :param rows: The list of names for the plate rows.
    :type columns: list of str
    :param columns: The list of names for the plate columns.
    :type wells: list of str or dict
    :param wells: The list of paths for the well groups.
    :type fmt: :class:`ome_zarr.format.Format`, optional
    :param fmt:
      The format of the ome_zarr data which should be used.
      Defaults to the most current.
    :type acquisitions: list of dict, optional
    :param acquisitions: A list of the various plate acquisitions.
    :type name: str, optional
    :param name: The plate name.
    :type field_count: int, optional
    :param field_count: The maximum number of fields per view across wells.
    """

    group, fmt = check_group_fmt(group, fmt)
    plate: dict[str, str | int | list[dict]] = {
        "columns": _validate_plate_rows_columns(columns),
        "rows": _validate_plate_rows_columns(rows),
        "wells": _validate_plate_wells(wells, rows, columns, fmt=fmt),
    }
    if name is not None:
        plate["name"] = name
    if field_count is not None:
        plate["field_count"] = field_count
    if acquisitions is not None:
        plate["acquisitions"] = _validate_plate_acquisitions(acquisitions)

    if fmt.version in ("0.1", "0.2", "0.3", "0.4"):
        plate["version"] = fmt.version
        group.attrs["plate"] = plate
    else:
        # Zarr v3 metadata under 'ome' with top-level version
        if fmt.version == "0.5":
            # See https://github.com/ome-zarr-models/ome-zarr-models-py/issues/218
            plate["version"] = fmt.version
        group.attrs["ome"] = {"version": fmt.version, "plate": plate}


def write_well_metadata(
    group: zarr.Group | str,
    images: list[str | dict],
    fmt: Format | None = None,
) -> None:
    """
    Write the well metadata in the group.

    :type group: :class:`zarr.Group`
    :param group: The zarr group or path to write the metadata in.
    :type images: list of dict
    :param images: The list of dictionaries for all fields of views.
    :type fmt: :class:`ome_zarr.format.Format`, optional
    :param fmt:
      The format of the ome_zarr data which should be used.
      Defaults to the most current.
    """

    group, fmt = check_group_fmt(group, fmt)
    well: dict[str, Any] = {
        "images": _validate_well_images(images),
    }

    if fmt.version in ("0.1", "0.2", "0.3", "0.4"):
        well["version"] = fmt.version
        group.attrs["well"] = well
    else:
        # Zarr v3 metadata under 'ome' with top-level version
        group.attrs["ome"] = {"version": fmt.version, "well": well}


def write_image(
    image: ArrayLike,
    group: zarr.Group | str,
    scale_factors: list[int] | tuple[int, ...] | list[dict[str, int]] | None = (
        2,
        4,
        8,
        16,
    ),
    name: str = "image",
    method: Methods | None = Methods.RESIZE,
    scaler: Scaler | None = None,
    fmt: Format | None = None,
    axes: AxesType = None,
    coordinate_transformations: list[list[dict[str, Any]]] | None = None,
    storage_options: JSONDict | list[JSONDict] | None = None,
    compute: bool = True,
    scale: dict[str, float] | None = None,
    axes_units: dict[str, str] | None = None,
    **metadata: JSONDict,
) -> list:
    """
    Write an image to the zarr store according to the OME-Zarr specification, supporting multiscale pyramids.

    Parameters
    ----------
    image : numpy.ndarray or dask.array.Array
        The image data to save. A downsampling pyramid will be computed if
        `scale_factors` is provided. Image array MUST be up to 5-dimensional with
        dimensions ordered (t, c, z, y, x). Can be a NumPy or Dask array.
    group : zarr.Group or str
        The zarr group to write the metadata, or a path to create
    scale: dict of str to float, optional
        The physical pixel size for each spatial dimension, e.g. {"z": 0.5, "y": 0.1, "x": 0.1}.
        If unset, the used pixel sizes default to 1.0 for all dimensions.
    scale_factors : Sequence[int] | list[dict[str, int]], optional
        The downsampling factors for each pyramid level. Default: (2, 4, 8, 16).
        Passing a list of integers (i.e., [2, 4, 8]) will apply the downsampling in all
        spatial dimensions *except the z dimension*, which will be left at a scale factor of 1.
        To apply downsampling to the z-dimension, pass the scale factors as a list of dicts, e.g.
        `[{"z": 2, "y": 2, "x": 2}, {"z": 4, "y": 4, "x": 4}, {"z": 8, "y": 8, "x": 8}]`.
        If dimensions are omitted in this dictionary,
        the downsampling factor for that dimension will default to 1.
    name: str, optional
        The name of the image, to be included in the metadata. Defaults to "image".
    axes_units : dict of str to str, optional
        The physical units for each dimension,
        e.g. {"t": "millisecond", "z": "micrometer", "y": "micrometer", "x": "micrometer"}.
        For a list of recommended units, see [ngff specification](https://ngff.openmicroscopy.org/specifications/0.5/index.html#axes-metadata).
    method : ome_zarr.scale.Methods, optional
        Downsampling method to use.
        Available methods are:
        - `nearest`: Nearest neighbor downsampling.
        - `resize`: Resize-based downsampling using `skimage.transform.resize` with anti-aliasing (default).
        - `laplacian`: Laplacian pyramid downsampling using `skimage.transform.pyramid_laplacian`.
        - `local_mean`: Local mean downsampling using `skimage.transform.downscale_local_mean`.
        - `zoom`: Zoom-based downsampling using `scipy.ndimage.zoom`.
    scaler : ome_zarr.scale.Scaler, optional
        [DEPRECATED] Scaler implementation for downsampling the image. Passing this
        argument will raise a warning and is no longer supported. Use `scale_factors` and
        `method` instead.
    fmt : ome_zarr.format.Format, optional
        The format of the ome_zarr data which should be used. Defaults to the most current.
    axes : list of str or list of dicts, optional
        The names of the axes, e.g. ["t", "c", "z", "y", "x"]. Ignored for versions 0.1 and 0.2.
        Required for version 0.3 or greater.
    coordinate_transformations : list of list of dict, optional
        [Deprecated] For each resolution, a list of transformation dicts (not validated).
        Each list of dicts is added to each dataset in order.
    storage_options : dict or list of dict, optional
        Options to be passed on to the storage backend. A list must match the number of datasets
        in a multiresolution pyramid. Allows different chunk sizes for each level.
        Regarding the key, value pairs in the dictionar(y)(ies), these depend both on the zarr_format used
        for writing and the dask version being used. For dask version <=2025.11.0, please refer to
        https://zarr.readthedocs.io/en/stable/api/zarr/create/#zarr.create for arguments that can be passed on.
        For >=2026.3.0 and up, please refer to https://zarr.readthedocs.io/en/stable/api/zarr/create/#zarr.create_array.
        It might be that you have to adjust the version of the docs. Note that the docs will also mention the
        differences of allowed arguments between zarr_format 2 and 3.

        Note: for chunks the default of `auto` is not allowed. This because the argument here refers to zarr chunks and
        autochunking here can result in different chunks then for the dask array. This can cause inconsistent overlap
        between dask and zarr chunks, potentially resulting in corrupted data. The default will be that if no sharding
        is specified, that the chunks correspond to the dask chunksize. This is also the case when chunks are provided as
        `None` and no sharding is provided.
    compute : bool, optional
        If True, compute immediately; otherwise, return a list of dask.delayed.Delayed objects.
    scale : dict of str to float, optional
        The physical pixel size for each dimension, e.g. {"z": 0.1, "y": 0.1, "x": 0.5}.
    axes_units : dict of str to str, optional
        The physical units for each dimension,
        e.g. {"t": "millisecond", "z": "micrometer", "y": "micrometer", "x": "micrometer"}.
        For a list of recommended units, see [ngff specification](https://ngff.openmicroscopy.org/specifications/0.5/index.html#axes-metadata).
    `**metadata` : dict
        Additional metadata to store, i.e. {"omero": {...}}. This is passed through to the multiscales metadata.

    Returns
    -------
    list
        Empty list if `compute` is True, otherwise a list of dask.delayed.Delayed objects
        representing the value to be computed by dask.

    Notes
    -----
    The `scaler` argument is deprecated and will be removed in a future version. Use
    `scale_factors` and `method` for all new code.
    """
    from .classes import OMEZarrImage, OMEZarrMultiscale

    if method is None:
        method = Methods.RESIZE

    group, fmt = check_group_fmt(group, fmt)

    if not isinstance(image, da.Array):
        image = da.from_array(image)

    if type(fmt) in (FormatV01, FormatV02, FormatV03):
        raise DeprecationWarning(
            f"Writing ome-zarr v{fmt.version} is deprecated and has been removed in version 0.15.0."
        )

    axes = _get_valid_axes(len(image.shape), axes, axes_units=axes_units, fmt=fmt)
    dims = _extract_dims_from_axes(axes)

    if scale is None:
        scale = dict.fromkeys(dims, 1.0)

    if coordinate_transformations is not None:
        msg = (
            "The 'coordinate_transformations' argument is deprecated and will "
            "be removed in a future version. Please use the `scale` argument "
            "to specify the physical pixel size for each dimension instead. "
        )
        warnings.warn(msg, DeprecationWarning)

    # parse scale_factors
    # if scaler is provided, we ignore scale_factors and infer the scale_factors
    # from the Scaler attributes instead.
    # for path, data in enumerate(pyramid):
    if scaler is not None:
        msg = """
            The 'scaler' argument is deprecated and will be removed in a future version.
            Please use the 'scale_factors' argument instead.
            """
        warnings.warn(msg, DeprecationWarning)

        scale_factors = [
            {d: 2 ** i if d in ("y", "x") else 1 for d in dims}
            for i in range(1, scaler.max_layer + 1)
        ]
        if scaler.method == "local_mean":
            method = Methods.LOCAL_MEAN
        elif scaler.method == "nearest":
            method = Methods.NEAREST
        elif scaler.method == "resize_image":
            method = Methods.RESIZE
        elif scaler.method == "laplacian":
            method = Methods.RESIZE
            warnings.warn(
                "Laplacian downsampling is not supported anymore."
                "Falling back to `resize`",
                UserWarning,
            )
        elif scaler.method == "zoom":
            method = Methods.ZOOM
        else:
            method = Methods.RESIZE

    omero = metadata.get("omero")

    singlescale = OMEZarrImage(
        data=image, scale=scale, axes=dims, name=name, axes_units=axes_units
    )
    multiscale = OMEZarrMultiscale(
        image=singlescale,
        scale_factors=scale_factors,
        method=method,
    )
    multiscale.omero = omero

    dask_delayed_jobs = multiscale.to_ome_zarr(
        group=group,
        storage_options=storage_options,
        version=cast(Literal["0.6.dev4", "0.5", "0.4"], fmt.version),
        compute=compute,
        overwrite=True,
    )

    return dask_delayed_jobs


def _resolve_storage_options(
    storage_options: JSONDict | list[JSONDict] | None, path: int
) -> JSONDict:
    options = {}
    if storage_options:
        options = (
            storage_options.copy()
            if not isinstance(storage_options, list)
            else storage_options[path]
        )
    return options


def _write_pyramid_to_zarr(
    pyramid: list[da.Array],
    group: zarr.Group,
    fmt: Format,
    scale: dict[str, float],
    axes: list[str] | tuple[str],
    axes_units: dict[str, str] | None = None,
    coordinate_transformations: list[list[dict[str, Any]]] | None = None,
    storage_options: JSONDict | list[JSONDict] | None = None,
    name: str | None = None,
    compute: bool = True,
    **metadata: str | JSONDict | list[JSONDict],
) -> list:

    group, fmt = check_group_fmt(group, fmt)

    # make sure every axis is represented in `scale`;
    # coerce to 1.0 if not provided
    # but don't allow missing axes to avoid silent errors
    scale = {d: scale.get(d, 1.0) for d in axes}

    # Set up common kwargs for da.to_zarr
    # zarr_array_kwargs needs dask 2025.12.0 or later
    zarr_array_kwargs: dict[str, Any] = {}
    zarr_format = zarr_array_kwargs["zarr_format"] = fmt.zarr_format
    options = _resolve_storage_options(storage_options, 0)

    if USE_DASK_ARRAY_KWARGS:
        if zarr_format == 2:
            zarr_array_kwargs["chunk_key_encoding"] = {"name": "v2", "separator": "/"}

        if "compressor" in options:
            # We use 'compressors' for group.create_array() but da.to_zarr() below uses
            # zarr.create() which doesn't support 'compressors'
            # TypeError: AsyncArray._create() got an unexpected keyword argument 'compressors'
            # kwargs["compressors"] = [options.pop("compressor", _blosc_compressor())]

            # ValueError: compressor cannot be used for arrays with zarr_format 3.
            # Use bytes-to-bytes codecs instead.
            zarr_array_kwargs["compressors"] = options.pop("compressor")
    elif zarr_format == 2:
        zarr_array_kwargs["dimension_separator"] = "/"

    if axes is not None and zarr_format != 2:
        zarr_array_kwargs["dimension_names"] = axes

    shapes = []
    datasets: list[dict] = []
    delayed = []

    for idx, level in enumerate(pyramid):
        zarr_array_kwargs_copy = zarr_array_kwargs.copy()
        options = _resolve_storage_options(storage_options, idx)
        if USE_DASK_ARRAY_KWARGS:
            options.pop("compressor", None)
        else:
            zarr_array_kwargs_copy["compressor"] = options.pop("compressor", None)

        # ensure that the chunk dimensions match the image dimensions
        # (which might have been changed for versions 0.1 or 0.2)
        # if chunks are explicitly set in the storage options
        if "compressors" not in zarr_array_kwargs_copy and USE_DASK_ARRAY_KWARGS:
            zarr_array_kwargs_copy["compressors"] = options.pop("compressors", "auto")

        chunks_opt = options.get("chunks", None)
        shards_opt = options.get("shards", None)

        # If shards are defined, one dask chunk should correspond to 1 shard to prevent concurrent writes to 1 shard.
        # In this case user defined chunks will correspond to zarr chunks and not dask chunks.
        # Check against string is purely because of mypy
        if chunks_opt and not isinstance(chunks_opt, str) and not shards_opt:
            chunks_opt = _retuple(chunks_opt, level.shape)
            level_image = da.array(level).rechunk(chunks=chunks_opt)
        elif shards_opt is not None:
            # This ensures that shards are always divisible by chunks, which is a requirement.
            if chunks_opt and chunks_opt != "auto":
                chunks_opt = _retuple(chunks_opt, level.shape)  # type: ignore[arg-type]
            else:
                # Technically not needed as ultimately in this case dask chunks will correspond to shards.
                # Simply adding this warning here to make the user used to not using "auto".
                if chunks_opt == "auto":
                    warnings.warn(
                        f"Setting `chunks` to `auto` is not allowed. Defaulting to the chunksize "
                        f"of dask array: {level.chunksize}."
                    )
                chunks_opt = level.chunksize
            chunks_opt = _retuple(chunks_opt, level.shape)
            shards_opt = _retuple(shards_opt, level.shape)
            level_image = da.array(level).rechunk(shards_opt)
        else:
            if chunks_opt == "auto":
                warnings.warn(
                    f"Setting `chunks` to `auto` is not allowed. Defaulting to the chunksize "
                    f"of dask array: {level.chunksize}."
                )
            chunks_opt = level.chunksize
            level_image = level

        shapes.append(level_image.shape)

        LOGGER.debug(
            "write dask.array to_zarr shape: %s, dtype: %s",
            level_image.shape,
            level_image.dtype,
        )

        zarr_array_kwargs_copy["chunks"] = chunks_opt
        zarr_array_kwargs_copy["shards"] = shards_opt

        for k, v in options.items():
            if k not in zarr_array_kwargs_copy:
                zarr_array_kwargs_copy[k] = v

        if not USE_DASK_ARRAY_KWARGS:
            if "chunks" in zarr_array_kwargs_copy:
                level_image = level_image.rechunk(zarr_array_kwargs_copy["chunks"])
                del zarr_array_kwargs_copy["chunks"]
            if zarr_format != 2:
                # zarr.create only allows compressor for zarr format 2, for 3 bytes-to-bytes codecs should be used.
                zarr_array_kwargs_copy["compressor"] = "auto"

            # Possibly non-exhaustive list of arguments not supported for zarr.create used by dask <=2025.11.0
            zarr_array_kwargs_copy.pop("compressors", None)
            zarr_array_kwargs_copy.pop("shards", None)
            zarr_array_kwargs_copy.pop("serializer", None)

        delayed.append(
            da.to_zarr(
                arr=level_image,
                url=group.store,
                component=str(Path(group.path, f"s{idx}")),
                compute=False,
                **zarr_array_kwargs_copy,
            )
        )
        datasets.append({"path": f"s{idx}"})

    # Computing delayed jobs if necessary
    if compute:
        da.compute(*delayed)
        delayed = []

    return delayed


def get_metadata(group: zarr.Group | str) -> dict:
    if isinstance(group, str):
        group = zarr.open_group(group, mode="r")
    attrs = group.attrs

    if group.info._zarr_format == 3:
        attrs = attrs.get("ome", {})
    else:
        attrs = dict(attrs)
    return attrs


def add_metadata(
    group: zarr.Group | str, metadata: JSONDict, fmt: Format | None = None
) -> None:

    group, fmt = check_group_fmt(group, fmt)

    attrs = group.attrs
    if fmt.version not in ("0.1", "0.2", "0.3", "0.4"):
        attrs = attrs.get("ome", {})

    for key, value in metadata.items():
        # merge dicts...
        if isinstance(value, dict) and isinstance(attrs.get(key), dict):
            attrs[key].update(value)
        else:
            attrs[key] = value

    if fmt.version in ("0.1", "0.2", "0.3", "0.4"):
        for key, value in attrs.items():
            group.attrs[key] = value
    else:
        # Zarr v3 metadata under 'ome' with top-level version
        group.attrs["ome"] = attrs


def write_multiscale_labels(
    pyramid: list,
    group: zarr.Group | str,
    name: str,
    fmt: Format | None = None,
    axes: AxesType = None,
    coordinate_transformations: list[list[dict[str, Any]]] | None = None,
    storage_options: JSONDict | list[JSONDict] | None = None,
    label_metadata: JSONDict | None = None,
    scale: dict[str, float] | None = None,
    axes_units: dict[str, str] | None = None,
    compute: bool = True,
) -> list:
    """
        Write precomputed pyramidal image labels to disk.

        This function writes a multiscale pyramid of label data to a zarr store,
        along with the appropriate metadata according to the OME-Zarr specification.
        The label data is saved under a `labels/{name}` subgroup at the specified `group` location.

        Parameters
        ----------
        pyramid : list of numpy.ndarray
            The image label data to save. The largest level should be first in the list.
            All image arrays MUST be up to 5-dimensional with dimensions ordered (t, c,
            z, y, x).
        group : zarr.Group or str
            The zarr group or path to write the data in.
            The label data will be saved under a `labels/{name}` subgroup.
        name : str
            The name of this labels data.
        fmt : ome_zarr.format.Format, optional
            The format of the ome_zarr data which should be used.
            Defaults to the most current.
        axes : list of str or list of dicts, optional
            The names of the axes, e.g. ["t", "c", "z", "y", "x"].
        axes_units : dict of str to str, optional
            The physical units for each dimension, e.g. {"t": "millisecond", "
    z": "micrometer", "y": "micrometer", "x": "micrometer"}.
            For a list of recommended units,
            see [ngff specification](https://ngff.openmicroscopy.org/specifications/0.5/index.html#axes-metadata).
        coordinate_transformations : list of list of dict, optional
            [DEPRECATED] For each resolution, a list of transformation dicts (not validated).
        storage_options : dict or list of dict, optional
            Options to be passed on to the storage backend.
            A list would need to match the number of datasets in a multiresolution pyramid.
            One can provide different chunk size for each level of a pyramid using this
            option.
            Regarding the key, value pairs in the dictionar(y)(ies), these depend both on the zarr_format used
            for writing and the dask version being used. For dask version <=2025.11.0, please refer to
            https://zarr.readthedocs.io/en/stable/api/zarr/create/#zarr.create for arguments that can be passed on.
            For >=2026.3.0 and up, please refer to https://zarr.readthedocs.io/en/stable/api/zarr/create/#zarr.create_array.
            It might be that you have to adjust the version of the docs. Note that the docs will also mention the
            differences of allowed arguments between zarr_format 2 and 3.

            Note: for chunks the default of `auto` is not allowed. This because the argument here refers to zarr chunks and
            autochunking here can result in different chunks then for the dask array. This can cause inconsistent overlap
            between dask and zarr chunks, potentially resulting in corrupted data. The default will be that if no sharding
            is specified, that the chunks correspond to the dask chunksize. This is also the case when chunks are provided as
            `None` and no sharding is provided.
        label_metadata : dict, optional
            Image label metadata.
            See [ngff specification](https://ngff.openmicroscopy.org/specifications/0.5/index.html#labels-metadata) for details.
            If not passed, is computed from the label data and stored in the metadata.
        scale : dict of str to float, optional
            The physical pixel size for each dimension, e.g. {"z": 0.1, "y": 0.1, "x": 0.5}.
            The pixel sizes for every passed resolution level are calculated directly from the defined `scale`
            for each resolution level. If not passed, defaults to 1.0 for all dimensions.
        axes_units : dict of str to str, optional
            The physical units for each axis, e.g. {"z": "micrometer", "y": "micrometer", "x": "micrometer"}.
        compute : bool, optional
            If True, compute immediately; otherwise, return a list of dask.delayed.Delayed objects.
    """
    from ome_zarr import OMEZarrImage, OMEZarrLabels

    group, fmt = check_group_fmt(group, fmt)
    dims = len(pyramid[0].shape)
    axes = _get_valid_axes(dims, axes, axes_units=axes_units, fmt=fmt)

    if scale is None:
        _axes = _get_valid_axes(
            len(pyramid[0].shape), axes, axes_units=axes_units, fmt=fmt
        )
        scale = dict.fromkeys(_extract_dims_from_axes(_axes), 1.0)

    if coordinate_transformations is not None:
        msg = (
            "The 'coordinate_transformations' argument is deprecated and will "
            "be removed in a future version. Please use the `scale` argument "
            "to specify the physical pixel size for each dimension instead. "
            "When `coordinate_transformations` is provided, it takes "
            "precedence over `scale`, so `scale` is not applied. When "
            "`coordinate_transformations` is not provided, the pixel sizes "
            "for every resolution level are calculated from `scale` and "
            "`scale_factors`."
        )
        warnings.warn(msg, DeprecationWarning)

    sub_group = group.require_group(f"labels/{name}")

    images: list[OMEZarrImage] = []
    for level in pyramid:
        relative_factor = np.asarray(level.shape) / np.asarray(pyramid[0].shape)
        level_scale = {
            d: s / relative_factor[i] for i, (d, s) in enumerate(scale.items())
        }
        images.append(
            OMEZarrImage(
                data=level,
                scale=level_scale,
                axes=list(scale.keys()),
                name=name,
                axes_units=axes_units,
            )
        )

    ms = OMEZarrLabels(image=images)
    if label_metadata is not None:
        ms.image_label = label_metadata

    dask_delayed_jobs = ms.to_ome_zarr(
        group=sub_group,
        storage_options=storage_options,
        version=cast(Literal["0.6.dev4", "0.5", "0.4"], fmt.version),
        compute=compute,
        overwrite=True,
    )

    label_list = []
    if fmt.version in ("0.1", "0.2", "0.3", "0.4"):
        node_metadata = group["labels"].attrs
    else:
        node_metadata = group["labels"].attrs.get("ome", {})

    label_list = node_metadata.get("labels", [])
    label_list.append(name)

    if fmt.version in ("0.1", "0.2", "0.3", "0.4"):
        group["labels"].attrs["labels"] = label_list
    else:
        group["labels"].attrs["ome"] = {
            "version": fmt.version,
            "labels": label_list,
        }

    return dask_delayed_jobs


def write_labels(
    labels: np.ndarray | da.Array,
    group: zarr.Group | str,
    name: str = "labels",
    scaler: Scaler | None = None,
    scale_factors: list[int] | tuple[int, ...] | list[dict[str, int]] = (2, 4, 8, 16),
    method: Methods = Methods.NEAREST,
    fmt: Format | None = None,
    axes: AxesType = None,
    coordinate_transformations: list[list[dict[str, Any]]] | None = None,
    storage_options: JSONDict | list[JSONDict] | None = None,
    label_metadata: JSONDict | None = None,
    scale: dict[str, float] | None = None,
    axes_units: dict[str, str] | None = None,
    compute: bool = True,
    **metadata: JSONDict,
) -> list:
    """
    Write image label data to disk, including multiscale and image-label metadata.
    Creates the label data in the sub-group "labels/{name}".

    Parameters
    ----------
    labels : numpy.ndarray or dask.array.Array
        The label data to save. A downsampling pyramid will be computed if
        `scale_factors` is provided. Label array MUST be up to 5-dimensional with
        dimensions ordered (t, c, z, y, x).
    group : zarr.Group
        The group within the zarr store to write the metadata in.
    scale: dict of str to float, optional
        The physical pixel size for each dimension, e.g. {"z": 0.1, "y": 0.1, "x": 0.5}.
        The pixel sizes for every resolution level are calculated directly from the defined `scale` and
        `scale_factors` for each level.
    name : str
        The name of this labels data.
    scale: dict of str to float, optional
        The physical pixel size for each dimension, e.g. {"z": 0.1, "y": 0.1, "x": 0.5}.
        The pixel sizes for every resolution level are calculated directly from the defined `scale` and
        `scale_factors` for each level.
    scaler : ome_zarr.scale.Scaler, optional
        [DEPRECATED] Scaler implementation for downsampling the label data. Passing this
        argument will raise a warning and is no longer supported. Use `scale_factors` and
        `method` instead.
    scale_factors : Sequence[int] | tuple[int, ...] | list[dict[str, int]], optional
        The downsampling factors for each pyramid level. Default: (2, 4, 8, 16).
        Passing a list of integers (i.e., [2, 4, 8]) will apply the downsampling in all
        spatial dimensions *except the z dimension*, which will be left at a scale factor of 1.
        To apply downsampling to the z-dimension, pass the scale factors as a list of dicts, e.g.
        `[{"z": 1, "y": 2, "x": 2}, {"z": 1, "y": 4, "x": 4}, {"z": 1, "y": 8, "x": 8}]`.
        This default behavior may change in future versions.
    axes_units : dict of str to str, optional
        The physical units for each dimension,
        e.g. {"t": "millisecond", "z": "micrometer", "y": "micrometer", "x": "micrometer"}.
        For a list of recommended units,
        see [ngff specification](https://ngff.openmicroscopy.org/specifications/0.5/index.html#axes-metadata).
    method : ome_zarr.scale.Methods, optional
        Downsampling method to use. Default: Methods.NEAREST (recommended for labels).
        See also `ome_zarr.scale.Methods` for available methods.
    fmt : ome_zarr.format.Format, optional
        The format of the ome_zarr data which should be used. Defaults to the most current.
    axes : list of str or list of dicts, optional
        The names of the axes, e.g. ["t", "c", "z", "y", "x"]. Ignored for versions 0.1 and 0.2.
        Required for version 0.3 or greater.
    coordinate_transformations : list of list of dict, optional
        [DEPRECATED] For each resolution, a list of transformation dicts (not validated). Each list of dicts
        is added to each dataset in order. When provided, this metadata takes precedence over the
        `scale`-derived transformations, so `scale` is ignored.
    storage_options : dict or list of dict, optional
        Options to be passed on to the storage backend. A list must match the number of datasets
        in a multiresolution pyramid. Allows different chunk sizes for each level.
        Regarding the key, value pairs in the dictionar(y)(ies), these depend both on the zarr_format used
        for writing and the dask version being used. For dask version <=2025.11.0, please refer to
        https://zarr.readthedocs.io/en/stable/api/zarr/create/#zarr.create for arguments that can be passed on.
        For >=2026.3.0 and up, please refer to https://zarr.readthedocs.io/en/stable/api/zarr/create/#zarr.create_array.
        It might be that you have to adjust the version of the docs. Note that the docs will also mention the
        differences of allowed arguments between zarr_format 2 and 3.

        Note: for chunks the default of `auto` is not allowed. This because the argument here refers to zarr chunks and
        autochunking here can result in different chunks then for the dask array. This can cause inconsistent overlap
        between dask and zarr chunks, potentially resulting in corrupted data. The default will be that if no sharding
        is specified, that the chunks correspond to the dask chunksize. This is also the case when chunks are provided as
        `None` and no sharding is provided.
    label_metadata : dict, optional
        Image label metadata. See :meth:`write_label_metadata` for details.
    compute : bool, optional
        If True, compute immediately; otherwise, return a list of dask.delayed.Delayed objects.
    scale : dict of str to float, optional
        The physical pixel size for each dimension, e.g. {"z": 0.1, "y": 0.1, "x": 0.5}.
    axes_units : dict of str to str, optional
        The physical units for each dimension,
        e.g. {"t": "millisecond", "z": "micrometer", "y": "micrometer", "x": "micrometer"}.
        For a list of recommended units, see [ngff specification](https://ngff.openmicroscopy.org/specifications/0.5/index.html#axes-metadata).
    `**metadata` : dict
        Additional metadata to store, i.e. {"image-label": {...}}. This is passed through to the image-label metadata.

    Returns
    -------
    list
        Empty list if `compute` is True, otherwise a list of dask.delayed.Delayed objects
        representing the value to be computed by dask.

    Notes
    -----
    The `scaler` argument is deprecated and will be removed in a future version. Use
    `scale_factors` and `method` for all new code. Labels downsampling should avoid interpolation;
    nearest-neighbor is recommended.
    """
    from .classes import OMEZarrImage, OMEZarrLabels

    group, fmt = check_group_fmt(group, fmt)
    sub_group = group.require_group(f"labels/{name}")

    if type(fmt) in (FormatV01, FormatV02, FormatV03):
        raise DeprecationWarning(
            f"Writing ome-zarr v{fmt.version} is deprecated and has been removed in version 0.15.0."
        )

    axes = _get_valid_axes(len(labels.shape), axes, axes_units=axes_units, fmt=fmt)
    dims = _extract_dims_from_axes(axes)

    if scale is None:
        scale = dict.fromkeys(dims, 1.0)

    if method is None:
        method = Methods.NEAREST

    if scaler is not None:
        msg = """
        The 'scaler' argument is deprecated and will be removed in version 0.13.0.
        Please use the 'scale_factors' argument instead.
        """
        scale_factors = [
            {d: 2 ** i if d in ("y", "x") else 1 for d in dims}
            for i in range(1, scaler.max_layer + 1)
        ]
        warnings.warn(msg, DeprecationWarning)

    if coordinate_transformations is not None:
        msg = (
            "The 'coordinate_transformations' argument is deprecated and will "
            "be removed in a future version. Please use the `scale` argument "
            "to specify the physical pixel size for each dimension instead. "
        )
        warnings.warn(msg, DeprecationWarning)

    singlescale = OMEZarrImage(
        data=labels, axes=dims, name=name, scale=scale, axes_units=axes_units
    )
    multiscales = OMEZarrLabels(
        image=singlescale,
        scale_factors=scale_factors,
        method=method,
    )

    label_metadata = metadata.get("image-label")
    if label_metadata is not None:
        multiscales.image_label = label_metadata

    dask_delayed_jobs = multiscales.to_ome_zarr(
        group=sub_group,
        storage_options=storage_options,
        version=cast(Literal["0.6.dev4", "0.5", "0.4"], fmt.version),
        compute=compute,
        overwrite=True,
    )

    label_list = []
    if fmt.version in ("0.1", "0.2", "0.3", "0.4"):
        node_metadata = group["labels"].attrs
    else:
        node_metadata = group["labels"].attrs.get("ome", {})

    label_list = node_metadata.get("labels", [])
    label_list.append(name)

    if fmt.version in ("0.1", "0.2", "0.3", "0.4"):
        group["labels"].attrs["labels"] = label_list
    else:
        group["labels"].attrs["ome"] = {
            "version": fmt.version,
            "labels": label_list,
        }

    return dask_delayed_jobs


def _retuple(chunks: tuple[Any, ...] | int, shape: tuple[Any, ...]) -> tuple[Any, ...]:
    """
    Expand chunks to match shape.

    E.g. if chunks is (64, 64) and shape is (3, 4, 5, 1028, 1028)
    return (3, 4, 5, 64, 64)

    If chunks is an integer, it is applied to all dimensions, to match
    the behaviour of zarr-python.
    """

    if isinstance(chunks, int):
        return tuple([chunks] * len(shape))

    dims_to_add = len(shape) - len(chunks)
    return (*shape[:dims_to_add], *chunks)
