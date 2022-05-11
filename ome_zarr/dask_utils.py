from typing import Tuple

import numpy as np
import skimage.transform
from dask import array as da

# This module contributed by Andreas Eisenbarth @aeisenbarth
# See https://github.com/toloudis/ome-zarr-py/pull/1


def resize(image: da.Array, output_shape: Tuple[int, ...], *args, **kwargs) -> da.Array:
    """
    Wrapped copy of "skimage.transform.resize"

    Resize image to match a certain size.

    Args:
        image: Input image.
        output_shape: Size of the generated output image
        *args: Arguments of skimage.transform.resize
        **kwargs: Keyword arguments of skimage.transform.resize

    Returns:
        Resized image.
    """
    factors = np.array(output_shape) / np.array(image.shape).astype(float)
    # Rechunk the input blocks so that the factors achieve an output blocks size of full numbers.
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

    Args:
        image: Input image.
        factors: Sequence of integers factors for each dimension.

    Returns:
        Resized image.
    """
    if not len(factors) == image.ndim:
        raise ValueError(
            f"Dimension mismatch: {image.ndim} image dimensions, {len(factors)} scale factors"
        )
    if not (
        all(isinstance(f, int) and 0 < f <= d for f, d in zip(factors, image.shape))
    ):
        raise ValueError(
            f"All scale factors must not be greater than the dimension length: ({tuple(factors)}) <= ({tuple(image.shape)})"
        )
    slices = tuple(slice(None, None, factor) for factor in factors)
    return image[slices]
