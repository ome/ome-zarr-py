"""Module for downsampling numpy arrays via various methods.

See the :class:`~ome_zarr.scale.Scaler` class for details.
"""

import inspect
import logging
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Union

import dask.array as da
import numpy as np
import zarr
from deprecated import deprecated
from scipy.ndimage import zoom
from skimage.transform import (
    downscale_local_mean,
    pyramid_gaussian,
    pyramid_laplacian,
    resize,
)

from .dask_utils import resize as dask_resize

LOGGER = logging.getLogger("ome_zarr.scale")

ListOfArrayLike = Union[list[da.Array], list[np.ndarray]]  # noqa: UP007  # FIXME
ArrayLike = Union[da.Array, np.ndarray]  # noqa: UP007  # FIXME


@deprecated(
    reason="Downsampling via the `Scaler` class has been deprecated. Please use the `scale_Factors` argument instead.",
    version="0.13.0",
)
@dataclass
class Scaler:
    """Helper class for performing various types of downsampling.

    A method can be chosen by name such as "nearest". All methods on this
    that do not begin with "_" and not either "methods" or "scale" are valid
    choices. These values can be returned by the
    :func:`~ome_zarr.scale.Scaler.methods` method.

    Attributes:
        copy_metadata:
            If `True`, copy Zarr attributes from the input array to the new group.
        downscale:
            Downscaling factor.
        in_place:
            Does not do anything.
        labeled:
            If `True`, check that the values in the downsampled levels are a subset
            of the values found in the input array.
        max_layer:
            The maximum number of downsampled layers to create.
        method:
            Downsampling method

    >>> import numpy as np
    >>> data = np.zeros((1, 1, 1, 64, 64))
    >>> scaler = Scaler()
    >>> downsampling = scaler.nearest(data)
    >>> for x in downsampling:
    ...     print(x.shape)
    (1, 1, 1, 64, 64)
    (1, 1, 1, 32, 32)
    (1, 1, 1, 16, 16)
    (1, 1, 1, 8, 8)
    (1, 1, 1, 4, 4)
    """

    copy_metadata: bool = False
    downscale: int = 2
    in_place: bool = False
    labeled: bool = False
    max_layer: int = 4
    method: str = "nearest"

    # 0: Nearest-neighbor
    # 1: Bi-linear (default)
    order: int = 1  # only used for resize

    @staticmethod
    def methods() -> Iterator[str]:
        """Return the name of all methods which define a downsampling.

        Any of the returned values can be used as the `methods`
        argument to the
        :func:`Scaler constructor <ome_zarr.scale.Scaler._init__>`
        """
        funcs = inspect.getmembers(Scaler, predicate=inspect.isfunction)
        for name, func in funcs:
            if name in ("methods", "scale"):
                continue
            if name.startswith("_"):
                continue
            yield name

    def scale(self, input_array: str, output_directory: str) -> None:
        """Perform downsampling to disk."""
        func = self.func

        # store = self.__check_store(output_directory)
        base = zarr.open_array(input_array)
        pyramid = func(base)

        if self.labeled:
            self.__assert_values(pyramid)

        grp = self.__create_group(output_directory, base, pyramid)

        if self.copy_metadata:
            print(f"copying attribute keys: {list(base.attrs.keys())}")
            grp.attrs.update(base.attrs)

    @property
    def func(self) -> Callable[[np.ndarray], list[np.ndarray]]:
        """Get downsample function."""
        func = getattr(self, self.method, None)
        if not func:
            raise Exception
        return func

    def __assert_values(self, pyramid: list[np.ndarray]) -> None:
        """Check for a single unique set of values for all pyramid levels."""
        expected = set(np.unique(pyramid[0]))
        print(f"level 0 {pyramid[0].shape} = {len(expected)} labels")
        for i in range(1, len(pyramid)):
            level = pyramid[i]
            print(f"level {i}", pyramid[i].shape, len(expected))
            found = set(np.unique(level))
            if not expected.issuperset(found):
                raise Exception(
                    f"{len(found)} found values are not "
                    "a subset of {len(expected)} values"
                )

    def __create_group(
        self, dir_path: str, base: np.ndarray, pyramid: list[np.ndarray]
    ) -> zarr.Group:
        """Create group and datasets."""
        grp = zarr.open_group(dir_path, mode="w")
        grp.create_dataset("base", data=base)
        series = []
        for i in range(len(pyramid)):
            if i == 0:
                path = "base"
            else:
                path = str(i)
                grp.create_dataset(path, data=pyramid[i])
            series.append({"path": path})
        return grp

    def resize_image(self, image: ArrayLike) -> ArrayLike:
        """
        Resize a numpy array OR a dask array to a smaller array (not pyramid)
        """
        if isinstance(image, da.Array):

            def _resize(image: ArrayLike, out_shape: tuple, **kwargs: Any) -> ArrayLike:
                return dask_resize(image, out_shape, **kwargs)

        else:
            _resize = resize

        # only down-sample in X and Y dimensions for now...
        new_shape = list(image.shape)
        new_shape[-1] = image.shape[-1] // self.downscale
        new_shape[-2] = image.shape[-2] // self.downscale
        out_shape = tuple(new_shape)

        dtype = image.dtype
        image = _resize(
            image.astype(float),
            out_shape,
            order=self.order,
            mode="reflect",
            anti_aliasing=False,
        )
        return image.astype(dtype)

    def nearest(self, base: np.ndarray) -> list[np.ndarray]:
        """
        Downsample using :func:`skimage.transform.resize`.
        """
        return self._by_plane(base, self.__nearest)

    def __nearest(self, plane: ArrayLike, sizeY: int, sizeX: int) -> np.ndarray:
        """Apply the 2-dimensional transformation."""
        if isinstance(plane, da.Array):

            def _resize(
                image: ArrayLike, output_shape: tuple, **kwargs: Any
            ) -> ArrayLike:
                return dask_resize(image, output_shape, **kwargs)

        else:
            _resize = resize

        return _resize(
            plane,
            output_shape=(sizeY // self.downscale, sizeX // self.downscale),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(plane.dtype)

    def gaussian(self, base: np.ndarray) -> list[np.ndarray]:
        """Downsample using :func:`skimage.transform.pyramid_gaussian`."""
        return list(
            pyramid_gaussian(
                base,
                downscale=self.downscale,
                max_layer=self.max_layer,
                channel_axis=None,
            )
        )

    def laplacian(self, base: np.ndarray) -> list[np.ndarray]:
        """Downsample using :func:`skimage.transform.pyramid_laplacian`."""
        return list(
            pyramid_laplacian(
                base,
                downscale=self.downscale,
                max_layer=self.max_layer,
                channel_axis=None,
            )
        )

    def local_mean(self, base: np.ndarray) -> list[np.ndarray]:
        """Downsample using :func:`skimage.transform.downscale_local_mean`."""
        rv = [base]
        stack_dims = base.ndim - 2
        factors = (*(1,) * stack_dims, *(self.downscale, self.downscale))
        for i in range(self.max_layer):
            rv.append(downscale_local_mean(rv[-1], factors=factors).astype(base.dtype))
        return rv

    def zoom(self, base: np.ndarray) -> list[np.ndarray]:
        """Downsample using :func:`scipy.ndimage.zoom`."""
        rv = [base]
        print(base.shape)
        for i in range(self.max_layer):
            print(i, self.downscale)
            rv.append(zoom(base, self.downscale**i))
            print(rv[-1].shape)
        return list(reversed(rv))

    #
    # Helpers
    #

    def _by_plane(
        self,
        base: np.ndarray,
        func: Callable[[np.ndarray, int, int], np.ndarray],
    ) -> np.ndarray:
        """Loop over 3 of the 5 dimensions and apply the func transform."""

        rv = [base]
        for i in range(self.max_layer):
            stack_to_scale = rv[-1]
            shape_5d = (*(1,) * (5 - stack_to_scale.ndim), *stack_to_scale.shape)
            T, C, Z, Y, X = shape_5d

            # If our data is already 2D, simply resize and add to pyramid
            if stack_to_scale.ndim == 2:
                rv.append(func(stack_to_scale, Y, X))
                continue

            # stack_dims is any dims over 2D
            stack_dims = stack_to_scale.ndim - 2
            new_stack = None
            for t in range(T):
                for c in range(C):
                    for z in range(Z):
                        dims_to_slice = (t, c, z)[-stack_dims:]
                        # slice nd down to 2D
                        plane = stack_to_scale[(dims_to_slice)][:]
                        out = func(plane, Y, X)
                        # first iteration of loop creates the new nd stack
                        if new_stack is None:
                            zct_dims = shape_5d[:-2]
                            shape_dims = zct_dims[-stack_dims:]
                            new_stack = np.zeros(
                                (*shape_dims, out.shape[0], out.shape[1]),
                                dtype=base.dtype,
                            )
                        # insert resized plane into the stack at correct indices
                        new_stack[(dims_to_slice)] = out
            rv.append(new_stack)
        return rv


SPATIAL_DIMS = ("z", "y", "x")


class Methods(Enum):
    RESIZE = "resize"
    NEAREST = "nearest"


def _build_pyramid(
    image: da.Array | np.ndarray,
    scale_factors: list[int],
    dims: Sequence[str],
    method: str | Methods = "nearest",
    chunks: tuple[int, ...] | None | str = None,
) -> list[da.Array]:
    """
    Build a pyramid of downscaled images.

    Parameters
    ----------
    image : dask.array.Array or numpy.ndarray
        The input image to downscale.
    scale_factors : list of int
        The downscaling factors for each pyramid level.
    dims : sequence of str
        The dimension names corresponding to the image axes.
    method : str or Methods, optional
        The downsampling method to use. Options are "resize" or "nearest".
        Default is "nearest".
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

    images: list[da.Array] = [image]

    for idx, factor in enumerate(scale_factors):
        # Compute relative factor for this level
        if idx == 0:
            relative_factor = scale_factors[0]
        else:
            relative_factor = factor // scale_factors[idx - 1]

        # Build per-dimension factor (only spatial dims are downsampled)
        per_dim_factor = tuple(
            relative_factor if d in SPATIAL_DIMS else 1 for d in dims
        )

        # Calculate target shape, leave non-spatial dims unchanged
        target_shape = []
        for s, d, f in zip(images[-1].shape, dims, per_dim_factor):
            if d in SPATIAL_DIMS:
                if s // f == 0:
                    target_shape.append(1)
                    warnings.warn(
                        f"Dimension {d} is too small to downsample further.",
                        UserWarning,
                        stacklevel=3,
                    )
                else:
                    target_shape.append(int(s // f))
            else:
                target_shape.append(int(s))

        if method == Methods.RESIZE:
            new_image = dask_resize(images[-1], output_shape=target_shape)
        elif method == Methods.NEAREST:
            new_image = dask_resize(
                images[-1],
                output_shape=target_shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            )
        else:
            raise ValueError(f"Unknown downsampling method: {method}")

        images.append(new_image)

    return images
