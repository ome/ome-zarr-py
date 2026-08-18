"""Functions for generating synthetic data."""

from collections.abc import Callable
from typing import Literal, cast

import dask.array as da
import numpy as np
import zarr
from skimage import data
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.segmentation import clear_border

from ome_zarr import OMEZarrImage, OMEZarrLabels, OMEZarrMultiscale

from .format import CurrentFormat, Format

CHANNEL_DIMENSION = 1


def coins() -> tuple[OMEZarrMultiscale, OMEZarrLabels]:
    """
    Sample data from skimage.

    Returns
    -------
    image : OMEZarrMultiscale
        Image array.
    labels : OMEZarrLabels
        Label array.
    """
    from skimage.morphology import closing, footprint_rectangle, remove_small_objects

    # Thanks to Juan
    # https://gist.github.com/jni/62e07ddd135dbb107278bc04c0f9a8e7
    image = data.coins()[50:-50, 50:-50]
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, footprint_rectangle((4, 4)))
    cleared = remove_small_objects(clear_border(bw), max_size=20)
    label_image = np.asarray(label(cleared))
    chunks = [s // 8 if s > 8 else 1 for s in image.shape]

    img = OMEZarrImage(
        data=da.from_array(image, chunks=chunks), axes="yx", name="coins"
    )
    lbl = OMEZarrImage(
        data=da.from_array(label_image, chunks=chunks), axes="yx", name="coins"
    )

    img_ms = OMEZarrMultiscale(
        image=img,
        contrast_limits=[(0, 255)],
        channel_colors=["FF0000"],
    )
    lbl_ms = OMEZarrLabels(image=lbl)

    return img_ms, lbl_ms


def astronaut() -> tuple[OMEZarrMultiscale, OMEZarrLabels]:
    """
    Sample data from skimage.

    Returns
    -------
    pyramids :
        List of pyramid arrays.
    labels :
        List of labels.
    """
    astro = data.astronaut()
    red = astro[:, :, 0]
    green = astro[:, :, 1]
    blue = astro[:, :, 2]
    astro = np.array([red, green, blue])
    pixels = np.tile(astro, (1, 2, 2))

    shape = list(pixels.shape)
    _c, y, x = shape
    label = np.zeros((y, x), dtype=np.int8)
    make_circle(100, 100, 1, label[200:300, 200:300])
    make_circle(150, 150, 2, label[250:400, 250:400])

    chunks = [s // 8 if s > 8 else 1 for s in pixels.shape]
    chunks_labels = [s // 8 if s > 8 else 1 for s in label.shape]

    img = OMEZarrImage(
        data=da.from_array(pixels, chunks=chunks), axes="cyx", name="astronaut"
    )
    lbl = OMEZarrImage(
        data=da.from_array(label, chunks=chunks_labels),
        axes="yx",
        name="astronaut_labels",
    )

    img_ms = OMEZarrMultiscale(
        image=img,
        contrast_limits=[(0, 255), (0, 255), (0, 255)],
        channel_colors=["FF0000", "00FF00", "0000FF"],
        channel_names=["Red", "Green", "Blue"],
    )
    lbl_ms = OMEZarrLabels(image=lbl)

    return img_ms, lbl_ms


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

    mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < r**2
    target[mask] = value


def rgb_to_5d(pixels: np.ndarray) -> list:
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
    method: Callable[..., tuple[OMEZarrMultiscale, OMEZarrLabels]] = coins,
    label_name: str = "coins",
    fmt: Format = CurrentFormat(),
    chunks: tuple | list | None = None,
) -> zarr.Group:
    """Generate a synthetic image pyramid with labels."""
    image, label = method()
    image.labels = {label.name: label}
    version = cast(Literal["0.4", "0.5", "0.6"], fmt.version)
    image.to_ome_zarr(
        zarr_directory,
        version=version,
        overwrite=True,
    )

    return zarr.open(zarr_directory, mode="a")
