"""Image writer utility

"""
import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import zarr

from .format import CurrentFormat, Format
from .scale import Scaler
from .types import JSONDict

LOGGER = logging.getLogger("ome_zarr.writer")

KNOWN_AXES = {"x": "space", "y": "space", "z": "space", "c": "channel", "t": "time"}


def _axes_to_dicts(
    axes: Union[List[str], List[Dict[str, str]]]
) -> List[Dict[str, str]]:
    """Returns a list of axis dicts with name and type"""
    axes_dicts = []
    for axis in axes:
        if isinstance(axis, str):
            axis_dict = {"name": axis}
            if axis in KNOWN_AXES:
                axis_dict["type"] = KNOWN_AXES[axis]
            axes_dicts.append(axis_dict)
        else:
            axes_dicts.append(axis)
    return axes_dicts


def _axes_to_names(axes: List[Dict[str, str]]) -> List[str]:
    """Returns a list of axis names"""
    axes_names = []
    for axis in axes:
        if "name" not in axis:
            raise ValueError("Axis Dict %s has no 'name'" % axis)
        axes_names.append(axis["name"])
    return axes_names


def _validate_axes_types(axes_dicts: List[Dict[str, str]]) -> None:
    """
    Validate the axes types according to the spec, version 0.4+
    """
    axes_types = [axis.get("type") for axis in axes_dicts]
    known_types = list(KNOWN_AXES.values())
    unknown_types = [atype for atype in axes_types if atype not in known_types]
    if len(unknown_types) > 1:
        raise ValueError(
            "Too many unknown axes types. 1 allowed, found: %s" % unknown_types
        )

    def _last_index(item: str, item_list: List[Any]) -> int:
        return max(loc for loc, val in enumerate(item_list) if val == item)

    if "time" in axes_types and _last_index("time", axes_types) > 0:
        raise ValueError("'time' axis must be first dimension only")

    if axes_types.count("channel") > 1:
        raise ValueError("Only 1 axis can be type 'channel'")

    if "channel" in axes_types and _last_index(
        "channel", axes_types
    ) > axes_types.index("space"):
        raise ValueError("'space' axes must come after 'channel'")


def _validate_axes(
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

    # axes may be list of 'x', 'y' or list of {'name': 'x'}
    axes_dicts = _axes_to_dicts(axes)
    axes_names = _axes_to_names(axes_dicts)

    # check names (only enforced for version 0.3)
    if fmt.version == "0.3":
        _validate_axes_03(axes_names)
        return axes_names

    _validate_axes_types(axes_dicts)

    return axes_dicts


def _validate_axes_03(axes: List[str]) -> None:

    val_axes = tuple(axes)
    if len(val_axes) == 2:
        if val_axes != ("y", "x"):
            raise ValueError(f"2D data must have axes ('y', 'x') {val_axes}")
    elif len(val_axes) == 3:
        if val_axes not in [("z", "y", "x"), ("c", "y", "x"), ("t", "y", "x")]:
            raise ValueError(
                "3D data must have axes ('z', 'y', 'x') or ('c', 'y', 'x')"
                " or ('t', 'y', 'x'), not %s" % (val_axes,)
            )
    elif len(val_axes) == 4:
        if val_axes not in [
            ("t", "z", "y", "x"),
            ("c", "z", "y", "x"),
            ("t", "c", "y", "x"),
        ]:
            raise ValueError("4D data must have axes tzyx or czyx or tcyx")
    else:
        if val_axes != ("t", "c", "z", "y", "x"):
            raise ValueError("5D data must have axes ('t', 'c', 'z', 'y', 'x')")


def write_multiscale(
    pyramid: List,
    group: zarr.Group,
    chunks: Union[Tuple[Any, ...], int] = None,
    fmt: Format = CurrentFormat(),
    axes: Union[str, List[str], List[Dict[str, str]]] = None,
) -> None:
    """
    Write a pyramid with multiscale metadata to disk.

    Parameters
    ----------
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
    """

    dims = len(pyramid[0].shape)
    axes = _validate_axes(dims, axes, fmt)

    paths = []
    for path, dataset in enumerate(pyramid):
        # TODO: chunks here could be different per layer
        group.create_dataset(str(path), data=dataset, chunks=chunks)
        paths.append(str(path))
    write_multiscales_metadata(group, paths, fmt, axes)


def write_multiscales_metadata(
    group: zarr.Group,
    paths: List[str],
    fmt: Format = CurrentFormat(),
    axes: Union[str, List[str], List[Dict[str, str]]] = None,
) -> None:
    """
    Write the multiscales metadata in the group.

    Parameters
    ----------
    group: zarr.Group
      the group within the zarr store to write the metadata in.
    paths: list of str
      The list of paths to the datasets for this multiscale image.
    fmt: Format
      The format of the ome_zarr data which should be used.
      Defaults to the most current.
    axes: list of str
      the names of the axes. e.g. ["t", "c", "z", "y", "x"].
      Ignored for versions 0.1 and 0.2. Required for version 0.3 or greater.
    """

    multiscales = [
        {
            "version": fmt.version,
            "datasets": [{"path": str(p)} for p in paths],
        }
    ]
    if axes is not None:
        if fmt.version in ("0.1", "0.2"):
            LOGGER.info("axes ignored for version 0.1 or 0.2")
        else:
            axes = _validate_axes(axes=axes, fmt=fmt)
            if axes is not None:
                multiscales[0]["axes"] = axes
    group.attrs["multiscales"] = multiscales


def write_image(
    image: np.ndarray,
    group: zarr.Group,
    chunks: Union[Tuple[Any, ...], int] = None,
    byte_order: Union[str, List[str]] = "tczyx",
    scaler: Scaler = Scaler(),
    fmt: Format = CurrentFormat(),
    axes: Union[str, List[str], List[Dict[str, str]]] = None,
    **metadata: JSONDict,
) -> None:
    """Writes an image to the zarr store according to ome-zarr specification

    Parameters
    ----------
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
    _validate_axes(image.ndim, axes, fmt)

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

    write_multiscale(image, group, chunks=chunks, fmt=fmt, axes=axes)
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
