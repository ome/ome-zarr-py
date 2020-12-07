"""Image writer utility

"""
import json
import logging
from pathlib import Path
from typing import List, Tuple, Union
from urllib.parse import urljoin

import dask.array as da
import numpy as np
import zarr

from .io import parse_url
from .reader import Node
from .types import JSONDict

LOGGER = logging.getLogger("ome_zarr.writer")


def write_image(
    path: str,
    image: np.ndarray,
    name: str = "0",
    group: str = None,
    chunks: Union[Tuple[int], int] = None,
    byte_order: Union[str, List[str]] = "tczyx",
    **metadata: JSONDict,
) -> zarr.hierarchy.Group:
    """Writes an image to the zarr store according to ome-zarr specification

    Parameters
    ----------
    path: str,
      a path to the zarr store location
    image: np.ndarray
      the image to save
    group: str, optional
      the group within the zarr store to store the data in
    chunks: int or tuple of ints,
      size of the saved chunks to store the image
    byte_order: str or list of str, default "tczyx"
      combination of the letters defining the order
      in which the dimensions are saved

    Return
    ------
    Zarr Group which contains the image.
    """

    zarr_location = parse_url(path, "w")
    node = Node(zarr=zarr_location, root=[])

    if image.ndim > 5:
        raise ValueError("Only images of 5D or less are supported")

    shape_5d = (*(1,) * (5 - image.ndim), *image.shape)
    image = image.reshape(shape_5d)

    if chunks is None:
        image = da.from_array(image)
    else:
        if isinstance(chunks, int):
            chunks = (chunks,)
        chunks = (*shape_5d[: (5 - len(chunks))], *chunks)
        image = da.from_array(image, chunks=chunks)

    omero = metadata.get("omero", {})

    # Update the size entry anyway
    omero["size"] = {
        "t": image.shape[0],
        "c": image.shape[1],
        "z": image.shape[2],
        "height": image.shape[3],
        "width": image.shape[4],
    }

    metadata["omero"] = omero
    da.to_zarr(arr=image, url=node.zarr.subpath(name))
    with open(Path(node.zarr.subpath(name)) / ".zattrs", "w") as za:
        json.dump(metadata, za)

    return node
