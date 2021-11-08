"""Image writer utility

"""
import logging
from typing import Any, List, Tuple, Union

import numpy as np
import zarr

from .format import CurrentFormat, Format
from .scale import Scaler
from .types import JSONDict

LOGGER = logging.getLogger("ome_zarr.writer")


def _validate_axes_names(
    ndim: int, axes: Union[str, List[str]] = None, fmt: Format = CurrentFormat()
) -> Union[None, List[str]]:
    """Returns validated list of axes names or raise exception if invalid"""

    if fmt.version in ("0.1", "0.2"):
        if axes is not None:
            LOGGER.info("axes ignored for version 0.1 or 0.2")
        return None

    # handle version 0.3...
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

    if isinstance(axes, str):
        axes = list(axes)

    if axes is not None:
        if len(axes) != ndim:
            raise ValueError("axes length must match number of dimensions")
        # from https://github.com/constantinpape/ome-ngff-implementations/
        val_axes = tuple(axes)
        if ndim == 2:
            if val_axes != ("y", "x"):
                raise ValueError(f"2D data must have axes ('y', 'x') {val_axes}")
        elif ndim == 3:
            if val_axes not in [("z", "y", "x"), ("c", "y", "x"), ("t", "y", "x")]:
                raise ValueError(
                    "3D data must have axes ('z', 'y', 'x') or ('c', 'y', 'x')"
                    " or ('t', 'y', 'x'), not %s" % (val_axes,)
                )
        elif ndim == 4:
            if val_axes not in [
                ("t", "z", "y", "x"),
                ("c", "z", "y", "x"),
                ("t", "c", "y", "x"),
            ]:
                raise ValueError("4D data must have axes tzyx or czyx or tcyx")
        else:
            if val_axes != ("t", "c", "z", "y", "x"):
                raise ValueError("5D data must have axes ('t', 'c', 'z', 'y', 'x')")

    return axes


def write_multiscale(
    pyramid: List,
    group: zarr.Group,
    chunks: Union[Tuple[Any, ...], int] = None,
    fmt: Format = CurrentFormat(),
    axes: Union[str, List[str]] = None,
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
    axes: str or list of str
      the names of the axes. e.g. "tczyx". Not needed for v0.1 or v0.2
      or for v0.3 if 2D or 5D. Otherwise this must be provided
    """

    dims = len(pyramid[0].shape)
    axes = _validate_axes_names(dims, axes, fmt)

    paths = []
    for path, dataset in enumerate(pyramid):
        # TODO: chunks here could be different per layer
        group.create_dataset(str(path), data=dataset, chunks=chunks)
        paths.append({"path": str(path)})

    multiscales = [{"version": fmt.version, "datasets": paths}]
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
    axes: Union[str, List[str]] = None,
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
    axes: str or list of str
      the names of the axes. e.g. "tczyx". Not needed for v0.1 or v0.2
      or for v0.3 if 2D or 5D. Otherwise this must be provided
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
    _validate_axes_names(image.ndim, axes, fmt)

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
