"""
Methods for downsampling numpy arrays via various methods.
"""

import inspect
import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Union, cast

import dask.array as da
import numpy as np
import zarr
from deprecated import deprecated
from scipy import __version__ as scipy_version
from scipy.ndimage import zoom
from skimage import __version__ as skimage_version
from skimage.transform import (
    downscale_local_mean,
    pyramid_gaussian,
    pyramid_laplacian,
    resize,
)

from .dask_utils import local_mean as dask_local_mean
from .dask_utils import resize as dask_resize
from .dask_utils import zoom as dask_zoom

LOGGER = logging.getLogger("ome_zarr.scale")

ListOfArrayLike = Union[list[da.Array], list[np.ndarray]]  # noqa: UP007  # FIXME
ArrayLike = Union[da.Array, np.ndarray]  # noqa: UP007  # FIXME

SPATIAL_DIMS = ("z", "y", "x")


class Methods(Enum):
    """Downsampling methods for multi-scale image generation.

    Each method uses different algorithms and parameters. Refer to the
    `method_dispatch` dictionary for detailed configuration of each approach.
    """

    RESIZE = "resize"
    """Bilinear interpolation downsampling (default).

    Uses :func:`skimage.transform.resize` with:

    - order=1 (bilinear interpolation)
    - mode='reflect'
    - anti_aliasing=True
    - preserve_range=True
    """

    NEAREST = "nearest"
    """Nearest-neighbor downsampling.

    Uses :func:`skimage.transform.resize` with:

    - order=0 (nearest neighbor)
    - mode='reflect'
    - anti_aliasing=False
    - preserve_range=True
    """

    LOCAL_MEAN = "local_mean"
    """Local mean downsampling.

    Uses :func:`skimage.transform.downscale_local_mean` to average pixel values
    in fixed-size neighborhoods.
    """

    ZOOM = "zoom"
    """Zoom-based flexible downsampling.

    Uses :func:`scipy.ndimage.zoom` for general-purpose downsampling.
    """


method_dispatch = {
    Methods.RESIZE: {
        "func": dask_resize,
        "kwargs": {
            "order": 1,
            "mode": "reflect",
            "anti_aliasing": True,
            "preserve_range": True,
        },
        "used_function": "skimage.transform.resize",
        "version": skimage_version,
    },
    Methods.NEAREST: {
        "func": dask_resize,
        "kwargs": {
            "order": 0,
            "mode": "reflect",
            "anti_aliasing": False,
            "preserve_range": True,
        },
        "used_function": "skimage.transform.resize",
        "version": skimage_version,
    },
    Methods.LOCAL_MEAN: {
        "func": dask_local_mean,
        "kwargs": {},
        "used_function": "skimage.transform.downscale_local_mean",
        "version": skimage_version,
    },
    Methods.ZOOM: {
        "func": dask_zoom,
        "kwargs": {},
        "used_function": "scipy.ndimage.zoom",
        "version": scipy_version,
    },
}


def _build_pyramid(
    image: da.Array | np.ndarray,
    scale_factors: list[dict[str, int]] | list[int] | tuple[int, ...],
    dims: Sequence[str],
    method: str | Methods = "nearest",
    chunks: tuple[int, ...] | str | None = None,
) -> list[da.Array]:
    """
    Build a pyramid of downscaled images.

    Parameters
    ----------
    image : dask.array.Array or numpy.ndarray
        The input image to downscale.
    scale_factors : list of dict or list of int or tuple of int
        The downscaling factors for each pyramid level. Each dictionary should
        map dimension names to their respective downscaling factors.
        I.e., [{"y": 2, "x": 2}, {"y": 4, "x": 4}] to create two levels with downsampling factors of 2 and 4.
        Alternatively, a list or tuple of integers can be provided, which will be
        converted to dictionaries with spatial dimensions only.
    dims : sequence of str
        The dimension names corresponding to the image axes.
    method : str or Methods, optional
        The downsampling method to use. Options are "resize" or "nearest".
        Default is "nearest".
    exact : bool, optional
        Whether to use exact resizing. Only applicable for certain methods.
        Default is False.
    chunks : tuple of int, str, or None, optional
        The chunk size to use for dask arrays. If None, the array's existing
        chunking is used. If a string, it should be a valid dask chunking
        specification. Default is None.
    """

    if isinstance(image, np.ndarray):
        if chunks is not None:
            image = da.from_array(image, chunks=chunks)
        else:
            image = da.from_array(image)

    if isinstance(method, str):
        method = Methods(method)

    # scale_factors are passed as [2, 4, 8, 16, ...]
    if isinstance(scale_factors, list | tuple) and all(
        isinstance(s, int) for s in scale_factors
    ):
        scales: list[dict[str, int]] = []
        for i in range(1, len(scale_factors) + 1):
            scale = {d: 2**i if d in SPATIAL_DIMS else 1 for d in dims}
            if "z" in dims:
                scale["z"] = 1
            scales.append(scale)
        scale_factors = scales
    else:
        scale_factors = cast(list[dict[str, int]], scale_factors)

    # Make sure scale_factors are represented in the same order as dims
    # and that dimensions that haven't been explicitly passed via the `scale_factors` argument
    # are set to a default of 1 (i.e. not downsampled)
    for i, level in enumerate(scale_factors):
        scale_factors[i] = {d: level.get(d, 1) for d in dims}

    images: list[da.Array] = [image]

    for idx, factor in enumerate(scale_factors):
        # Compute relative factor for this level
        # May or may not be an integer depending on passed scale factors
        if idx == 0:
            relative_factors = np.asarray(list(scale_factors[0].values()))
        else:
            relative_factors = np.asarray(list(factor.values())) / np.asarray(
                list(scale_factors[idx - 1].values())
            )

        # Calculate target shape, leave non-spatial dims unchanged
        target_shape = []
        for s, d, f in zip(images[-1].shape, dims, relative_factors):
            if d in SPATIAL_DIMS:
                if s // f == 0:
                    target_shape.append(1)
                    LOGGER.warning(
                        "Dimension %s is too small to downsample further.", d
                    )
                else:
                    target_shape.append(int(s // f))
            else:
                target_shape.append(int(s))

        if method not in method_dispatch:
            raise ValueError(f"Unknown downsampling method: {method}")

        # apply the downsampling method to the last image in the list
        new_image = method_dispatch[method]["func"](
            images[-1],
            output_shape=tuple(target_shape),
            **method_dispatch[method]["kwargs"],
        )

        images.append(new_image)

    return images
