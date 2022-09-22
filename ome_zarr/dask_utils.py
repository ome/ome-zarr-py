from typing import Any, Tuple

import numpy as np
import skimage.transform
from dask import array as da

# This module contributed by Andreas Eisenbarth @aeisenbarth
# See https://github.com/toloudis/ome-zarr-py/pull/1


def resize(
    image: da.Array, output_shape: Tuple[int, ...], *args: Any, **kwargs: Any
) -> da.Array:
    r"""
    Wrapped copy of "skimage.transform.resize"
    Resize image to match a certain size.
    :type image: :class:`dask.array`
    :param image: The dask array to resize
    :type output_shape: tuple
    :param output_shape: The shape of the resize array
    :type \*args: list
    :param \*args: Arguments of skimage.transform.resize
    :type \*\*kwargs: dict
    :param \*\*kwargs: Keyword arguments of skimage.transform.resize
    :return: Resized image.
    """
    factors = np.array(output_shape) / np.array(image.shape).astype(float)
    # Rechunk the input blocks so that the factors achieve an output
    # blocks size of full numbers.
    better_chunksize = tuple(
        np.maximum(1, np.round(np.array(image.chunksize) * factors) / factors).astype(
            int
        )
    )
    image_prepared = image.rechunk(better_chunksize)
    block_output_shape = tuple(
        np.floor(np.array(better_chunksize) * factors).astype(int)
    )

    # Map overlap
    def resize_block(image_block: da.Array, block_info: dict) -> da.Array:
        return skimage.transform.resize(
            image_block, block_output_shape, *args, **kwargs
        ).astype(image_block.dtype)

    output_slices = tuple(slice(0, d) for d in output_shape)
    output = da.map_blocks(
        resize_block, image_prepared, dtype=image.dtype, chunks=block_output_shape
    )[output_slices]
    return output.rechunk(image.chunksize).astype(image.dtype)


def downscale_nearest(image: da.Array, factors: Tuple[int, ...]) -> da.Array:
    """
    Primitive downscaling by integer factors using stepped slicing.
    :type image: :class:`dask.array`
    :param image: The dask array to resize
    :type factors: tuple
    :param factors: Sequence of integers factors for each dimension.
    :return: Resized image.
    """
    if not len(factors) == image.ndim:
        raise ValueError(
            f"Dimension mismatch: {image.ndim} image dimensions, "
            f"{len(factors)} scale factors"
        )
    if not (
        all(isinstance(f, int) and 0 < f <= d for f, d in zip(factors, image.shape))
    ):
        raise ValueError(
            f"All scale factors must not be greater than the dimension length: "
            f"({tuple(factors)}) <= ({tuple(image.shape)})"
        )
    slices = tuple(slice(None, None, factor) for factor in factors)
    return image[slices]
