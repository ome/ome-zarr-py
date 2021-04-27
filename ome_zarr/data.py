"""Functions for generating synthetic data."""
from random import randrange
from typing import Callable, List, Tuple

import numpy as np
import zarr
from scipy.ndimage import zoom
from skimage import data
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import closing, remove_small_objects, square
from skimage.segmentation import clear_border

from .io import parse_url
from .scale import Scaler
from .writer import write_multiscale

CHANNEL_DIMENSION = 1


def coins() -> Tuple[List, List]:
    """Sample data from skimage."""
    # Thanks to Juan
    # https://gist.github.com/jni/62e07ddd135dbb107278bc04c0f9a8e7
    image = data.coins()[50:-50, 50:-50]
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(4))
    cleared = remove_small_objects(clear_border(bw), 20)
    label_image = label(cleared)

    pyramid = list(reversed([zoom(image, 2 ** i, order=3) for i in range(4)]))
    labels = list(reversed([zoom(label_image, 2 ** i, order=0) for i in range(4)]))

    pyramid = [rgb_to_5d(layer) for layer in pyramid]
    labels = [rgb_to_5d(layer) for layer in labels]
    return pyramid, labels


def astronaut() -> Tuple[List, List]:
    """Sample data from skimage."""
    scaler = Scaler()

    pixels = rgb_to_5d(np.tile(data.astronaut(), (2, 2, 1)))
    pyramid = scaler.nearest(pixels)

    shape = list(pyramid[0].shape)
    shape[1] = 1
    label = np.zeros(shape)
    make_circle(100, 100, 1, label[0, 0, 0, 200:300, 200:300])
    make_circle(150, 150, 2, label[0, 0, 0, 250:400, 250:400])
    labels = scaler.nearest(label)

    return pyramid, labels


def make_circle(h: int, w: int, value: int, target: np.ndarray) -> None:
    """Apply a 2D circular mask to the given array.

    >>> import numpy as np
    >>> example = np.zeros((8, 8))
    >>> make_circle(8, 8, 1, example)
    >>> print(example)
    [[0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 1. 1. 1. 1. 1. 0.]
     [0. 1. 1. 1. 1. 1. 1. 1.]
     [0. 1. 1. 1. 1. 1. 1. 1.]
     [0. 1. 1. 1. 1. 1. 1. 1.]
     [0. 1. 1. 1. 1. 1. 1. 1.]
     [0. 1. 1. 1. 1. 1. 1. 1.]
     [0. 0. 1. 1. 1. 1. 1. 0.]]
    """
    x = np.arange(0, w)
    y = np.arange(0, h)

    cx = w // 2
    cy = h // 2
    r = min(w, h) // 2

    mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < r ** 2
    target[mask] = value


def rgb_to_5d(pixels: np.ndarray) -> List:
    """Convert an RGB image into 5D image (t, c, z, y, x)."""
    if len(pixels.shape) == 2:
        stack = np.array([pixels])
        channels = np.array([stack])
    elif len(pixels.shape) == 3:
        size_c = pixels.shape[2]
        channels = [np.array([pixels[:, :, c]]) for c in range(size_c)]
    else:
        assert False, f"expecting 2 or 3d: ({pixels.shape})"
    video = np.array([channels])
    return video


def create_zarr(
    zarr_directory: str,
    method: Callable[..., Tuple[List, List]] = coins,
    label_name: str = "coins",
) -> None:
    """Generate a synthetic image pyramid with labels."""
    pyramid, labels = method()

    loc = parse_url(zarr_directory, mode="w")
    assert loc
    grp = zarr.group(loc.store)
    write_multiscale(pyramid, grp)

    if pyramid[0].shape[CHANNEL_DIMENSION] == 1:
        image_data = {
            "channels": [{"window": {"start": 0, "end": 1}}],
            "rdefs": {"model": "greyscale"},
        }
    else:
        image_data = {
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
    grp.attrs["omero"] = image_data

    if labels:

        labels_grp = grp.create_group("labels")
        labels_grp.attrs["labels"] = [label_name]

        label_grp = labels_grp.create_group(label_name)
        write_multiscale(labels, label_grp)

        colors = []
        properties = []
        for x in range(1, 9):
            rgba = [randrange(0, 256) for i in range(4)]
            colors.append({"label-value": x, "rgba": rgba})
            properties.append({"label-value": x, "class": f"class {x}"})
        label_grp.attrs["image-label"] = {
            "version": "0.2",
            "colors": colors,
            "properties": properties,
            "source": {"image": "../../"},
        }
