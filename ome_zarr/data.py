#!/usr/bin/env python
from typing import List, Tuple

import numpy as np
import zarr
from scipy.ndimage import zoom
from skimage import data
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import closing, remove_small_objects, square
from skimage.segmentation import clear_border


def coins() -> Tuple[List, List]:
    """
    Sample data from skimage
    """
    # Thanks to Juan
    # https://gist.github.com/jni/62e07ddd135dbb107278bc04c0f9a8e7
    image = data.coins()[50:-50, 50:-50]
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(4))
    cleared = remove_small_objects(clear_border(bw), 20)
    label_image = label(cleared)
    pyramid = list(reversed([zoom(image, 2 ** i, order=3) for i in range(4)]))
    labels = list(reversed([zoom(label_image, 2 ** i, order=0) for i in range(4)]))
    return pyramid, labels


def rgba_to_int(r: int, g: int, b: int, a: int) -> int:
    return int.from_bytes([r, g, b, a], byteorder="big", signed=True)


def rgb_to_5d(pixels: np.ndarray) -> List:
    """convert an RGB image into 5D image (t, c, z, y, x)"""
    if len(pixels.shape) == 2:
        channels = [[np.array(pixels)]]
    elif len(pixels.shape) == 3:
        size_c = pixels.shape(2)
        channels = [np.array(pixels[:, :, c]) for c in range(size_c)]
    else:
        assert f"expecting 2 or 3d: ({pixels.shape})"
    return [np.array(channels)]


def write_multiscale(pyramid: List, group: zarr.Group) -> None:

    paths = []
    for path, dataset in enumerate(pyramid):
        group.create_dataset(str(path), data=pyramid[path])
        paths.append({"path": str(path)})

    multiscales = [{"version": "0.1", "datasets": paths}]
    group.attrs["multiscales"] = multiscales


def create_zarr(zarr_directory: str) -> None:

    pyramid, labels = coins()
    pyramid = [rgb_to_5d(layer) for layer in pyramid]
    labels = [rgb_to_5d(layer) for layer in labels]

    store = zarr.DirectoryStore(zarr_directory)
    grp = zarr.group(store)
    write_multiscale(pyramid, grp)

    labels_grp = grp.create_group("labels")
    labels_grp.attrs["labels"] = ["coins"]

    image_data = {
        "id": 1,
        "channels": [
            {
                "color": "FF0000",
                "window": {"start": 0, "end": 1},
                "label": "Red",
                "active": True,
            },
            {
                "color": "00FF00",
                "window": {"start": 0, "end": 1},
                "label": "Green",
                "active": True,
            },
            {
                "color": "0000FF",
                "window": {"start": 0, "end": 1},
                "label": "Blue",
                "active": True,
            },
        ],
        "rdefs": {"model": "color"},
    }
    if False:  # FIXME
        grp.attrs["omero"] = image_data

    coins_grp = labels_grp.create_group("coins")
    write_multiscale(labels, coins_grp)
    coins_grp.attrs["color"] = {
        "1": rgba_to_int(50, 0, 0, 0),
        "2": rgba_to_int(0, 50, 0, 0),
        "3": rgba_to_int(0, 0, 50, 0),
        "4": rgba_to_int(100, 0, 0, 0),
        "5": rgba_to_int(0, 100, 0, 0),
        "6": rgba_to_int(0, 0, 100, 0),
        "7": rgba_to_int(50, 50, 50, 0),
        "8": rgba_to_int(100, 100, 100, 0),
    }
    coins_grp.attrs["image"] = {"array": "../../", "source": {}}
