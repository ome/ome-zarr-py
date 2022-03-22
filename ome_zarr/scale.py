"""Module for downsampling numpy arrays via various methods.

See the :class:`~ome_zarr.scale.Scaler` class for details.
"""
import inspect
import logging
import os
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Callable, Iterator, List

import numpy as np
import zarr
from scipy.ndimage import zoom
from skimage.transform import (
    downscale_local_mean,
    pyramid_gaussian,
    pyramid_laplacian,
    resize,
)

from .io import parse_url

LOGGER = logging.getLogger("ome_zarr.scale")


@dataclass
class Scaler:
    """Helper class for performing various types of downsampling.

    A method can be chosen by name such as "nearest". All methods on this
    that do not begin with "_" and not either "methods" or "scale" are valid
    choices. These values can be returned by the
    :func:`~ome_zarr.scale.Scaler.methods` method.

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
        func = getattr(self, self.method, None)
        if not func:
            raise Exception

        store = self.__check_store(output_directory)
        base = zarr.open_array(input_array)
        pyramid = func(base)

        if self.labeled:
            self.__assert_values(pyramid)

        grp = self.__create_group(store, base, pyramid)

        if self.copy_metadata:
            print(f"copying attribute keys: {list(base.attrs.keys())}")
            grp.attrs.update(base.attrs)

    def __check_store(self, output_directory: str) -> MutableMapping:
        """Return a Zarr store if it doesn't already exist."""
        assert not os.path.exists(output_directory)
        loc = parse_url(output_directory, mode="w")
        assert loc
        return loc.store

    def __assert_values(self, pyramid: List[np.ndarray]) -> None:
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
        self, store: MutableMapping, base: np.ndarray, pyramid: List[np.ndarray]
    ) -> zarr.hierarchy.Group:
        """Create group and datasets."""
        grp = zarr.group(store)
        grp.create_dataset("base", data=base)
        series = []
        for i, dataset in enumerate(pyramid):
            if i == 0:
                path = "base"
            else:
                path = "%s" % i
                grp.create_dataset(path, data=pyramid[i])
            series.append({"path": path})
        return grp

    def nearest(self, base: np.ndarray) -> List[np.ndarray]:
        """
        Downsample using :func:`skimage.transform.resize`.

        The :const:`cvs2.INTER_NEAREST` interpolation method is used.
        """
        return self._by_plane(base, self.__nearest)

    def __nearest(self, plane: np.ndarray, sizeY: int, sizeX: int) -> np.ndarray:
        """Apply the 2-dimensional transformation."""
        return resize(
            plane,
            output_shape=(sizeY // self.downscale, sizeX // self.downscale),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(plane.dtype)

    def gaussian(self, base: np.ndarray) -> List[np.ndarray]:
        """Downsample using :func:`skimage.transform.pyramid_gaussian`."""
        return list(
            pyramid_gaussian(
                base,
                downscale=self.downscale,
                max_layer=self.max_layer,
                multichannel=False,
            )
        )

    def laplacian(self, base: np.ndarray) -> List[np.ndarray]:
        """Downsample using :func:`skimage.transform.pyramid_laplacian`."""
        return list(
            pyramid_laplacian(
                base,
                downscale=self.downscale,
                max_layer=self.max_layer,
                multichannel=False,
            )
        )

    def local_mean(self, base: np.ndarray) -> List[np.ndarray]:
        """Downsample using :func:`skimage.transform.downscale_local_mean`."""
        rv = [base]
        stack_dims = base.ndim - 2
        factors = (*(1,) * stack_dims, *(self.downscale, self.downscale))
        for i in range(self.max_layer):
            rv.append(downscale_local_mean(rv[-1], factors=factors))
        return rv

    def zoom(self, base: np.ndarray) -> List[np.ndarray]:
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
