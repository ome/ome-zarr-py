"""Image writer utility"""

import logging
import warnings
from pathlib import Path
from typing import Any, TypeAlias

import dask
import dask.array as da
import numpy as np
import zarr
from dask.graph_manipulation import bind
from numcodecs import Blosc

from .axes import Axes
from .format import CurrentFormat, Format, FormatV04
from .scale import Scaler
from .types import JSONDict

LOGGER = logging.getLogger("ome_zarr.writer")

ListOfArrayLike = list[da.Array] | list[np.ndarray]
ArrayLike: TypeAlias = da.Array | np.ndarray

AxesType = str | list[str] | list[dict[str, str]] | None


def _get_valid_axes(
    ndim: int | None = None,
    axes: AxesType = None,
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
    axes_obj = Axes(axes, fmt)

    return axes_obj.to_list(fmt)


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
        if fmt is None:
            fmt = CurrentFormat()
        group = zarr.open_group(group, mode=mode, zarr_format=fmt.zarr_format)
    else:
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
    group: zarr.Group | str,
    chunks: tuple[Any, ...] | int | None = None,
    fmt: Format | None = None,
    axes: AxesType = None,
    coordinate_transformations: list[list[dict[str, Any]]] | None = None,
    storage_options: JSONDict | list[JSONDict] | None = None,
    name: str | None = None,
    compute: bool | None = True,
    **metadata: str | JSONDict | list[JSONDict],
) -> list:
    """
    Write a pyramid with multiscale metadata to disk.

    :type pyramid: list of :class:`numpy.ndarray` or :class:`dask.array.Array`
    :param pyramid:
        The image data to save. Largest level first. All image arrays MUST be up to
        5-dimensional with dimensions ordered (t, c, z, y, x)
    :type group: :class:`zarr.Group`
    :param group: The group within the zarr store to store the data in
    :type chunks: int or tuple of ints, optional
    :param chunks:
        The size of the saved chunks to store the image.

        .. deprecated:: 0.4.0
            This argument is deprecated and will be removed in a future version.
            Use :attr:`storage_options` instead.
    :type fmt: :class:`ome_zarr.format.Format`, optional
    :param fmt:
        The format of the ome_zarr data which should be used.
        Defaults to the most current.
    :type axes: str list of str or list of dict, optional
    :param axes:
        List of axes dicts, or names. Not needed for v0.1 or v0.2 or if 2D. Otherwise
        this must be provided
    :type coordinate_transformations: 2Dlist of dict, optional
    :param coordinate_transformations:
        List of transformations for each path.
        Each list of dicts are added to each datasets in order and must include a
        'scale' transform.
    :type storage_options: dict or list of dict, optional
    :param storage_options:
        Options to be passed on to the storage backend.
        A list would need to match the number of datasets in a multiresolution pyramid.
        One can provide different chunk size for each level of a pyramid using this
        option.
    :param compute:
        If true compute immediately otherwise a list of :class:`dask.delayed.Delayed`
        is returned.
    :return:
        Empty list if the compute flag is True, otherwise it returns a list of
        :class:`dask.delayed.Delayed` representing the value to be computed by
        dask.
    """
    group, fmt = check_group_fmt(group, fmt)
    dims = len(pyramid[0].shape)
    axes = _get_valid_axes(dims, axes, fmt)
    dask_delayed = []

    if chunks is not None:
        msg = """The 'chunks' argument is deprecated and will be removed in version 0.5.
Please use the 'storage_options' argument instead."""
        warnings.warn(msg, DeprecationWarning)
    datasets: list[dict] = []
    for path, data in enumerate(pyramid):
        options = _resolve_storage_options(storage_options, path)

        # ensure that the chunk dimensions match the image dimensions
        # (which might have been changed for versions 0.1 or 0.2)
        # if chunks are explicitly set in the storage options
        chunks_opt = options.pop("chunks", chunks)
        if chunks_opt is not None:
            chunks_opt = _retuple(chunks_opt, data.shape)

        options["chunk_key_encoding"] = fmt.chunk_key_encoding
        zarr_format = fmt.zarr_format
        compressor = options.pop("compressor", None)
        if zarr_format == 2:
            # by default we use Blosc with zstd compression
            # Don't need this for zarr v3 as it has a default compressor
            if compressor is None:
                compressor = _blosc_compressor()
            options["compressor"] = compressor
        else:
            if compressor is not None:
                options["compressors"] = [compressor]
            if axes is not None:
                # the array zarr.json also contains axes names
                # TODO: check if this is written by da.to_zarr
                options["dimension_names"] = [
                    axis["name"] for axis in axes if isinstance(axis, dict)
                ]

        if isinstance(data, da.Array):
            if zarr_format == 2:
                options["dimension_separator"] = "/"
                del options["chunk_key_encoding"]
            # handle any 'chunks' option from storage_options
            if chunks_opt is not None:
                data = da.array(data).rechunk(chunks=chunks_opt)  # noqa: PLW2901
            da_delayed = da.to_zarr(
                arr=data,
                url=group.store,
                component=str(Path(group.path, str(path))),
                compute=compute,
                zarr_format=zarr_format,
                **options,
            )

            if not compute:
                dask_delayed.append(da_delayed)

        else:
            if chunks_opt is not None:
                options["chunks"] = chunks_opt
            options["shape"] = data.shape
            # otherwise we get 'null'
            options["fill_value"] = 0

            arr = group.create_array(
                str(path),
                dtype=data.dtype,
                **options,
            )
            arr[slice(None)] = data

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

    if len(dask_delayed) > 0 and not compute:
        write_multiscales_metadata_delayed = dask.delayed(write_multiscales_metadata)
        return dask_delayed + [
            bind(write_multiscales_metadata_delayed, dask_delayed)(
                group, datasets, fmt, axes, name, **metadata
            )
        ]
    else:
        write_multiscales_metadata(group, datasets, fmt, axes, name, **metadata)

    return dask_delayed


def write_multiscales_metadata(
    group: zarr.Group | str,
    datasets: list[dict],
    fmt: Format | None = None,
    axes: AxesType = None,
    name: str | None = None,
    **metadata: str | JSONDict | list[JSONDict],
) -> None:
    """
    Write the multiscales metadata in the group.

    :type group: :class:`zarr.Group`
    :param group: The zarr group or path.
    :type datasets: list of dicts
    :param datasets:
      The list of datasets (dicts) for this multiscale image.
      Each dict must include 'path' and a 'coordinateTransformations'
      list for version 0.4 or later that must include a 'scale' transform.
    :type fmt: :class:`ome_zarr.format.Format`, optional
    :param fmt:
      The format of the ome_zarr data which should be used.
      Defaults to the most current.
    :type axes: list of str or list of dicts, optional
    :param axes:
      The names of the axes. e.g. ["t", "c", "z", "y", "x"].
      Ignored for versions 0.1 and 0.2. Required for version 0.3 or greater.
    """

    group, fmt = check_group_fmt(group, fmt)
    ndim = -1
    if axes is not None:
        if fmt.version in ("0.1", "0.2"):
            LOGGER.info("axes ignored for version 0.1 or 0.2")
            axes = None
        else:
            axes = _get_valid_axes(axes=axes, fmt=fmt)
            if axes is not None:
                ndim = len(axes)
    if (
        isinstance(metadata, dict)
        and metadata.get("metadata")
        and isinstance(metadata["metadata"], dict)
        and "omero" in metadata["metadata"]
    ):
        omero_metadata = metadata["metadata"].pop("omero")
        if omero_metadata is None:
            raise KeyError("If `'omero'` is present, value cannot be `None`.")
        for c in omero_metadata["channels"]:
            if "color" in c:  # noqa: SIM102
                if not isinstance(c["color"], str) or len(c["color"]) != 6:
                    raise TypeError("`'color'` must be a hex code string.")
            if "window" in c:
                if not isinstance(c["window"], dict):
                    raise TypeError("`'window'` must be a dict.")
                for p in ["min", "max", "start", "end"]:
                    if p not in c["window"]:
                        raise KeyError(f"`'{p}'` not found in `'window'`.")
                    if not isinstance(c["window"][p], (int, float)):
                        raise TypeError(f"`'{p}'` must be an int or float.")

        add_metadata(group, {"omero": omero_metadata})

    # note: we construct the multiscale metadata via dict(), rather than {}
    # to avoid duplication of protected keys like 'version' in **metadata
    # (for {} this would silently over-write it, with dict() it explicitly fails)
    multiscales = [
        dict(datasets=_validate_datasets(datasets, ndim, fmt), name=name or group.name)
    ]
    if len(metadata.get("metadata", {})) > 0:
        multiscales[0]["metadata"] = metadata["metadata"]
    if axes is not None:
        # multiscales[0]["axes"] = axes
        fmt.write_axes(multiscales[0], axes)

    if fmt.version in ("0.1", "0.2", "0.3", "0.4"):
        multiscales[0]["version"] = fmt.version
    else:
        # Zarr v3 top-level version
        add_metadata(group, {"version": fmt.version})

    add_metadata(group, {"multiscales": multiscales})


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
    scaler: Scaler = Scaler(),
    chunks: tuple[Any, ...] | int | None = None,
    fmt: Format | None = None,
    axes: AxesType = None,
    coordinate_transformations: list[list[dict[str, Any]]] | None = None,
    storage_options: JSONDict | list[JSONDict] | None = None,
    compute: bool | None = True,
    **metadata: str | JSONDict | list[JSONDict],
) -> list:
    """Writes an image to the zarr store according to ome-zarr specification

    :type image: :class:`numpy.ndarray` or `dask.array.Array`
    :param image:
      The image data to save. A downsampling of the data will be computed
      if the scaler argument is non-None.
      Image array MUST be up to 5-dimensional with dimensions
      ordered (t, c, z, y, x).  Image can be a numpy or dask Array.
    :type group: :class:`zarr.Group` or str
    :param group: The zarr group to write the metadata, or a path to create
    :type scaler: :class:`ome_zarr.scale.Scaler`
    :param scaler:
      Scaler implementation for downsampling the image argument. If None,
      no downsampling will be performed.
    :type chunks: int or tuple of ints, optional
    :param chunks:
        The size of the saved chunks to store the image.

        .. deprecated:: 0.4.0
            This argument is deprecated and will be removed in a future version.
            Use :attr:`storage_options` instead.
    :type fmt: :class:`ome_zarr.format.Format`, optional
    :param fmt:
      The format of the ome_zarr data which should be used.
      Defaults to the most current.
    :type axes: list of str or list of dicts, optional
    :param axes:
      The names of the axes. e.g. ["t", "c", "z", "y", "x"].
      Ignored for versions 0.1 and 0.2. Required for version 0.3 or greater.
    :type coordinate_transformations: list of dict
    :param coordinate_transformations:
      For each resolution, we have a List of transformation Dicts (not validated).
      Each list of dicts are added to each datasets in order.
    :type storage_options: dict or list of dict, optional
    :param storage_options:
        Options to be passed on to the storage backend.
        A list would need to match the number of datasets in a multiresolution pyramid.
        One can provide different chunk size for each level of a pyramid using this
        option.
    :param compute:
        If true compute immediately otherwise a list of :class:`dask.delayed.Delayed`
        is returned.
    :return:
        Empty list if the compute flag is True, otherwise it returns a list of
        :class:`dask.delayed.Delayed` representing the value to be computed by
        dask.
    """

    # if a path is given create the group according to the format (default current)
    # otherwise check the group and fmt are compatible zarr versions
    group, fmt = check_group_fmt(group, fmt)
    dask_delayed_jobs = []

    name = metadata.pop("name", None)
    name = str(name) if name is not None else None
    if isinstance(image, da.Array):
        dask_delayed_jobs = _write_dask_image(
            image,
            group,
            scaler,
            chunks=chunks,
            fmt=fmt,
            axes=axes,
            coordinate_transformations=coordinate_transformations,
            storage_options=storage_options,
            name=name,
            compute=compute,
            **metadata,
        )
    else:
        mip = _create_mip(image, fmt, scaler, axes)
        dask_delayed_jobs = write_multiscale(
            mip,
            group,
            chunks=chunks,
            fmt=fmt,
            axes=axes,
            coordinate_transformations=coordinate_transformations,
            storage_options=storage_options,
            name=name,
            compute=compute,
            **metadata,
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


def _write_dask_image(
    image: da.Array,
    group: zarr.Group | str,
    scaler: Scaler = Scaler(),
    chunks: tuple[Any, ...] | int | None = None,
    fmt: Format | None = None,
    axes: AxesType = None,
    coordinate_transformations: list[list[dict[str, Any]]] | None = None,
    storage_options: JSONDict | list[JSONDict] | None = None,
    name: str | None = None,
    compute: bool | None = True,
    **metadata: str | JSONDict | list[JSONDict],
) -> list:
    group, fmt = check_group_fmt(group, fmt)
    if fmt.version in ("0.1", "0.2"):
        # v0.1 and v0.2 are strictly 5D
        shape_5d: tuple[Any, ...] = (*(1,) * (5 - image.ndim), *image.shape)
        image = image.reshape(shape_5d)
        # and we don't need axes
        axes = None

    dims = len(image.shape)
    axes = _get_valid_axes(dims, axes, fmt)

    if chunks is not None:
        msg = """The 'chunks' argument is deprecated and will be removed in version 0.5.
Please use the 'storage_options' argument instead."""
        warnings.warn(msg, DeprecationWarning)

    datasets: list[dict] = []
    delayed = []

    # for path, data in enumerate(pyramid):
    max_layer: int = scaler.max_layer if scaler is not None else 0
    shapes = []
    for path in range(max_layer + 1):
        # LOGGER.debug(f"write_image path: {path}")
        options = _resolve_storage_options(storage_options, path)

        # don't downsample top level of pyramid
        if str(path) != "0" and scaler is not None:
            image = scaler.resize_image(image)

        # ensure that the chunk dimensions match the image dimensions
        # (which might have been changed for versions 0.1 or 0.2)
        # if chunks are explicitly set in the storage options
        chunks_opt = options.pop("chunks", chunks)
        # switch to this code in 0.5
        # chunks_opt = options.pop("chunks", None)
        if chunks_opt is not None:
            chunks_opt = _retuple(chunks_opt, image.shape)
            # image.chunks will be used by da.to_zarr
            image = da.array(image).rechunk(chunks=chunks_opt)
        LOGGER.debug("chunks_opt: %s", chunks_opt)
        shapes.append(image.shape)

        LOGGER.debug(
            "write dask.array to_zarr shape: %s, dtype: %s", image.shape, image.dtype
        )
        kwargs: dict[str, Any] = {}
        zarr_format = fmt.zarr_format
        if zarr_format == 2:
            kwargs["dimension_separator"] = "/"
            kwargs["compressor"] = options.pop("compressor", _blosc_compressor())
        else:
            kwargs["chunk_key_encoding"] = fmt.chunk_key_encoding
            if axes is not None:
                kwargs["dimension_names"] = [
                    a["name"] for a in axes if isinstance(a, dict)
                ]
            if "compressor" in options:
                # We use 'compressors' for group.create_array() but da.to_zarr() below uses
                # zarr.create() which doesn't support 'compressors'
                # TypeError: AsyncArray._create() got an unexpected keyword argument 'compressors'
                # kwargs["compressors"] = [options.pop("compressor", _blosc_compressor())]

                # ValueError: compressor cannot be used for arrays with zarr_format 3.
                # Use bytes-to-bytes codecs instead.
                kwargs["compressor"] = options.pop("compressor")

        delayed.append(
            da.to_zarr(
                arr=image,
                url=group.store,
                component=str(Path(group.path, str(path))),
                compute=False,
                zarr_format=zarr_format,
                **kwargs,
            )
        )
        datasets.append({"path": str(path)})

    # Computing delayed jobs if necessary
    if compute:
        da.compute(*delayed)
        delayed = []

    if coordinate_transformations is None:
        # shapes = [data.shape for data in delayed]
        coordinate_transformations = fmt.generate_coordinate_transformations(shapes)

    # we validate again later, but this catches length mismatch before zip(datasets...)
    fmt.validate_coordinate_transformations(
        dims, len(datasets), coordinate_transformations
    )
    if coordinate_transformations is not None:
        for dataset, transform in zip(datasets, coordinate_transformations):
            dataset["coordinateTransformations"] = transform
    if not compute:
        write_multiscales_metadata_delayed = dask.delayed(write_multiscales_metadata)
        return delayed + [
            bind(write_multiscales_metadata_delayed, delayed)(
                group, datasets, fmt, axes, name, **metadata
            )
        ]
    else:
        write_multiscales_metadata(group, datasets, fmt, axes, name, **metadata)
        return delayed


def write_label_metadata(
    group: zarr.Group | str,
    name: str,
    colors: list[JSONDict] | None = None,
    properties: list[JSONDict] | None = None,
    fmt: Format | None = None,
    **metadata: list[JSONDict] | JSONDict | str,
) -> None:
    """
    Write image-label metadata to the group.

    The label data must have been written to a sub-group,
    with the same name as the second argument.

    :type group: :class:`zarr.Group`
    :param group: The zarr group or path to write the metadata in.
    :type name: str
    :param name: The name of the label sub-group.
    :type colors: list of JSONDict, optional
    :param colors:
      Fixed colors for (a subset of) the label values.
      Each dict specifies the color for one label and must contain the fields
      "label-value" and "rgba".
    :type properties: list of JSONDict, optional
    :param properties:
      Additional properties for (a subset of) the label values.
      Each dict specifies additional properties for one label.
      It must contain the field "label-value"
      and may contain arbitrary additional properties.
    :type fmt: :class:`ome_zarr.format.Format`, optional
    :param fmt:
      The format of the ome_zarr data which should be used.
      Defaults to the most current.
    """
    group, fmt = check_group_fmt(group, fmt)
    label_group = group[name]
    image_label_metadata = {**metadata}
    if colors is not None:
        image_label_metadata["colors"] = colors
    if properties is not None:
        image_label_metadata["properties"] = properties
    image_label_metadata["version"] = fmt.version

    label_list = get_metadata(group).get("labels", [])
    label_list.append(name)

    add_metadata(group, {"labels": label_list}, fmt=fmt)
    add_metadata(label_group, {"image-label": image_label_metadata}, fmt=fmt)


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
    group: zarr.Group, metadata: JSONDict, fmt: Format | None = None
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
    chunks: tuple[Any, ...] | int | None = None,
    fmt: Format | None = None,
    axes: AxesType = None,
    coordinate_transformations: list[list[dict[str, Any]]] | None = None,
    storage_options: JSONDict | list[JSONDict] | None = None,
    label_metadata: JSONDict | None = None,
    compute: bool | None = True,
    **metadata: JSONDict,
) -> list:
    """
    Write pyramidal image labels to disk.

    Including the multiscales and image-label metadata.
    Creates the label data in the sub-group "labels/{name}"

    :type pyramid: list of :class:`numpy.ndarray`
    :param pyramid:
      the image label data to save. Largest level first
      All image arrays MUST be up to 5-dimensional with dimensions
      ordered (t, c, z, y, x)
    :type group: :class:`zarr.Group`
    :param group: The zarr group or path to write the metadata in.
    :type name: str, optional
    :param name: The name of this labels data.
    :type chunks: int or tuple of ints, optional
    :param chunks:
        The size of the saved chunks to store the image.

        .. deprecated:: 0.4.0
            This argument is deprecated and will be removed in a future version.
            Use :attr:`storage_options` instead.
    :type fmt: :class:`ome_zarr.format.Format`, optional
    :param fmt:
      The format of the ome_zarr data which should be used.
      Defaults to the most current.
    :type axes: list of str or list of dicts, optional
    :param axes:
      The names of the axes. e.g. ["t", "c", "z", "y", "x"].
      Ignored for versions 0.1 and 0.2. Required for version 0.3 or greater.
    :type coordinate_transformations: list of dict
    :param coordinate_transformations:
      For each resolution, we have a List of transformation Dicts (not validated).
      Each list of dicts are added to each datasets in order.
    :type storage_options: dict or list of dict, optional
    :param storage_options:
        Options to be passed on to the storage backend.
        A list would need to match the number of datasets in a multiresolution pyramid.
        One can provide different chunk size for each level of a pyramid using this
        option.
    :type label_metadata: dict, optional
    :param label_metadata:
      Image label metadata. See :meth:`write_label_metadata` for details
    :param compute:
        If true compute immediately otherwise a list of :class:`dask.delayed.Delayed`
        is returned.
    :return:
        Empty list if the compute flag is True, otherwise it returns a list of
        :class:`dask.delayed.Delayed` representing the value to be computed by
        dask.
    """
    group, fmt = check_group_fmt(group, fmt)
    sub_group = group.require_group(f"labels/{name}")
    dask_delayed_jobs = write_multiscale(
        pyramid,
        sub_group,
        chunks=chunks,
        fmt=fmt,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
        storage_options=storage_options,
        name=name,
        compute=compute,
        **metadata,
    )
    write_label_metadata(
        group["labels"],
        name,
        fmt=fmt,
        **({} if label_metadata is None else label_metadata),
    )

    return dask_delayed_jobs


def write_labels(
    labels: np.ndarray | da.Array,
    group: zarr.Group | str,
    name: str,
    scaler: Scaler = Scaler(order=0),
    chunks: tuple[Any, ...] | int | None = None,
    fmt: Format | None = None,
    axes: AxesType = None,
    coordinate_transformations: list[list[dict[str, Any]]] | None = None,
    storage_options: JSONDict | list[JSONDict] | None = None,
    label_metadata: JSONDict | None = None,
    compute: bool | None = True,
    **metadata: JSONDict,
) -> list:
    """
    Write image label data to disk.

    Including the multiscales and image-label metadata.
    Creates the label data in the sub-group "labels/{name}"

    :type labels: :class:`numpy.ndarray`
    :param labels:
      The label data to save. A downsampling of the data will be computed
      if the scaler argument is non-None.
      Label array MUST be up to 5-dimensional with dimensions
      ordered (t, c, z, y, x)
    :type group: :class:`zarr.Group` or str
    :param group: The zarr group or path to write the metadata in.
    :type name: str, optional
    :param name: The name of this labels data.
    :type scaler: :class:`ome_zarr.scale.Scaler`
    :param scaler:
      Scaler implementation for downsampling the image argument.
      NB: Labels downsampling should avoid interpolation.
      If no scaler is provided, the default scaler with nearest neighbour
      interpolation will be used.
      If scaler=None, no downsampling will be performed.
    :type chunks: int or tuple of ints, optional
    :param chunks:
        The size of the saved chunks to store the image.

        .. deprecated:: 0.4.0
            This argument is deprecated and will be removed in a future version.
            Use :attr:`storage_options` instead.
    :type fmt: :class:`ome_zarr.format.Format`, optional
    :param fmt:
      The format of the ome_zarr data which should be used.
      Defaults to the most current.
    :type axes: list of str or list of dicts, optional
    :param axes:
      The names of the axes. e.g. ["t", "c", "z", "y", "x"].
      Ignored for versions 0.1 and 0.2. Required for version 0.3 or greater.
    :type coordinate_transformations: list of dict
    :param coordinate_transformations:
      For each resolution, we have a List of transformation Dicts (not validated).
      Each list of dicts are added to each datasets in order.
    :type storage_options: dict or list of dict, optional
    :param storage_options:
        Options to be passed on to the storage backend.
        A list would need to match the number of datasets in a multiresolution pyramid.
        One can provide different chunk size for each level of a pyramid using this
        option.
    :type label_metadata: dict, optional
    :param label_metadata:
      Image label metadata. See :meth:`write_label_metadata` for details
    :param compute:
        If true compute immediately otherwise a list of :class:`dask.delayed.Delayed`
        is returned.
    :return:
        Empty list if the compute flag is True, otherwise it returns a list of
        :class:`dask.delayed.Delayed` representing the value to be computed by
        dask.
    """
    group, fmt = check_group_fmt(group, fmt)
    sub_group = group.require_group(f"labels/{name}")
    dask_delayed_jobs = []

    if isinstance(labels, da.Array):
        dask_delayed_jobs = _write_dask_image(
            labels,
            sub_group,
            scaler,
            chunks=chunks,
            fmt=fmt,
            axes=axes,
            coordinate_transformations=coordinate_transformations,
            storage_options=storage_options,
            name=name,
            compute=compute,
            **metadata,
        )
    else:
        mip = _create_mip(labels, fmt, scaler, axes)
        dask_delayed_jobs = write_multiscale(
            mip,
            sub_group,
            chunks=chunks,
            fmt=fmt,
            axes=axes,
            coordinate_transformations=coordinate_transformations,
            storage_options=storage_options,
            name=name,
            compute=compute,
            **metadata,
        )
    write_label_metadata(
        group=group["labels"],
        name=name,
        fmt=fmt,
        **({} if label_metadata is None else label_metadata),
    )

    return dask_delayed_jobs


def _create_mip(
    image: np.ndarray,
    fmt: Format,
    scaler: Scaler,
    axes: AxesType,
) -> list[np.ndarray]:
    """
    Generate a downsampled pyramid of images.

    Returns
    -------
    pyramid :
        List of numpy arrays that are the downsampled pyramid levels.
    """
    if image.ndim > 5:
        raise ValueError("Only images of 5D or less are supported")

    if fmt.version in ("0.1", "0.2"):
        # v0.1 and v0.2 are strictly 5D
        shape_5d: tuple[Any, ...] = (*(1,) * (5 - image.ndim), *image.shape)
        image = image.reshape(shape_5d)

    # check axes before trying to scale
    _get_valid_axes(image.ndim, axes, fmt)

    if scaler is not None:
        if image.shape[-1] == 1 or image.shape[-2] == 1:
            raise ValueError(
                "Can't downsample if size of x or y dimension is 1. "
                f"Shape: {image.shape}"
            )
        mip = scaler.func(image)
    else:
        LOGGER.debug("disabling pyramid")
        mip = [image]
    return mip


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
