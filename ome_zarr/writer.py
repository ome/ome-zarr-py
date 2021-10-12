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
    if fmt.version not in ("0.1", "0.2"):
        if axes is None:
            if dims == 2:
                axes = ["y", "x"]
            elif dims == 5:
                axes = ["t", "c", "z", "y", "x"]
            else:
                raise ValueError(
                    "axes must be provided. Can't be guessed for 3D or 4D data"
                )
        if len(axes) != dims:
            raise ValueError("axes length must match number of dimensions")

        if isinstance(axes, str):
            axes = list(axes)

        for dim in axes:
            if dim not in ("t", "c", "z", "y", "x"):
                raise ValueError("axes must each be one of 'x', 'y', 'z', 'c' or 't'")

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

    if chunks is not None:
        chunks = _retuple(chunks, image.shape)

    if scaler is not None:
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
