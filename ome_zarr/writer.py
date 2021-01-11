"""Image writer utility

"""
import logging
from typing import Any, List, Tuple, Union

import dask.array as da
import numpy as np
import zarr

from .types import JSONDict

LOGGER = logging.getLogger("ome_zarr.writer")


def write_multiscale(
    pyramid: List, group: zarr.Group, chunks: Union[Tuple[int], int] = None,
) -> None:
    """Write a pyramid with multiscale metadata to disk."""
    paths = []
    for path, dataset in enumerate(pyramid):
        # TODO: chunks here could be different per layer
        group.create_dataset(str(path), data=pyramid[path], chunks=chunks)
        paths.append({"path": str(path)})

    multiscales = [{"version": "0.1", "datasets": paths}]
    group.attrs["multiscales"] = multiscales


def write_image(
    image: np.ndarray,
    group: zarr.Group,
    chunks: Union[Tuple[int], int] = None,
    byte_order: Union[str, List[str]] = "tczyx",
    **metadata: JSONDict,
) -> None:
    """Writes an image to the zarr store according to ome-zarr specification

    Parameters
    ----------
    image: np.ndarray
      the image to save
    group: zarr.Group
      the group within the zarr store to store the data in
    chunks: int or tuple of ints,
      size of the saved chunks to store the image
    byte_order: str or list of str, default "tczyx"
      combination of the letters defining the order
      in which the dimensions are saved
    """

    if image.ndim > 5:
        raise ValueError("Only images of 5D or less are supported")

    shape_5d: Tuple[Any, ...] = (*(1,) * (5 - image.ndim), *image.shape)
    image = image.reshape(shape_5d)

    if chunks is not None:
        _chunks = _retuple(chunks, shape_5d)
        image = da.from_array(image, chunks=_chunks)

    omero = metadata.get("omero", {})

    # Update the size entry anyway
    omero["size"] = {
        "t": image.shape[0],
        "c": image.shape[1],
        "z": image.shape[2],
        "height": image.shape[3],
        "width": image.shape[4],
    }
    if omero.get("channels") is None:
        size_c = image.shape[1]
        if size_c == 1:
            omero["channels"] = [{"window": {"start": 0, "end": 1}}]
            omero["rdefs"] = {"model": "greyscale"}
        else:
            rng = np.random.default_rng(0)
            colors = rng.integers(0, high=2 ** 8, size=(image.shape[1], 3))
            omero["channels"] = [
                {
                    "color": "".join(f"{i:02x}" for i in color),
                    "window": {"start": 0, "end": 1},
                    "active": True,
                }
                for color in colors
            ]
            omero["rdefs"] = {"model": "color"}

    metadata["omero"] = omero
    write_multiscale([image], group)  # TODO: downsample
    group.attrs.update(metadata)


def _retuple(chunks: Union[Tuple[int], int], shape: Tuple[Any, ...]) -> Tuple[int, ...]:

    _chunks: Tuple[int]
    if isinstance(chunks, int):
        _chunks = (chunks,)
    else:
        _chunks = chunks

    return (*shape[: (5 - len(_chunks))], *_chunks)
