"""Image writer utility

"""
import logging

from typing import Any, Dict, Iterator, List, Optional, Type, Union, Tuple, cast
import zarr

import numpy as np

from .io import parse_url
from .reader import Node


LOGGER = logging.getLogger("ome_zarr.writer")

def write_image(
        path: str,
        image: np.ndarray,
        name: str = "0",
        group: str = None,
        chunks: Union[Tuple[int], int] = None,
        byte_order: Union[str, List[str]] = "tczyx",
        **metadata):
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
    """
    zarr_location = parse_url(path)

    ## I assume this should be dealt with in
    ## the ZarrLocation classes
    store = zarr.DirectoryStore(path)
    image = np.asarray(image)

    if image.ndim > 5:
        # Maybe we can split the more than 5D images in subroups?
        raise ValueError("Only images of 5D or less are supported")

    shape_5d = (*(1,)*(5 - image.ndim), *image.shape)
    image = image.reshape(shape_5d)
    omero = metadata.get("omero", {})

    # Update the size entry anyway
    omero["size"]= {
            "t": image.shape[0],
            "c": image.shape[1],
            "z": image.shape[2],
            "height": image.shape[3],
            "width": image.shape[4],
            }

    metadata["omero"] = omero

    root = zarr.group(store)
    if group is not None:
        grp = root.create_group(group)
    else:
        grp = root

    grp.create_dataset(name, data=image, chunks=chunks)

    for entry, value in metadata.items():
        grp.attrs[entry] = value
