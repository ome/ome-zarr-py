"""Module for downsampling numpy arrays via various methods.

See the :class:`~ome_zarr.scale.Scaler` class for details.
"""
import inspect
import logging
import os
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Callable, Iterator, List, Union

import dask.array as da
import numpy as np
import zarr
from scipy.ndimage import zoom
from skimage.transform import downscale_local_mean, pyramid_gaussian, pyramid_laplacian
from skimage.transform import resize as skimage_resize

from .dask_utils import resize as dask_resize
from .io import parse_url

LOGGER = logging.getLogger("ome_zarr.scale")

ListOfArrayLike = Union[List[da.Array], List[np.ndarray]]
ArrayLike = Union[da.Array, np.ndarray]


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

    def scale_array(self, input_array: ArrayLike) -> ListOfArrayLike:
        """Perform downsampling to memory."""
        func = getattr(self, self.method, None)
        if not func:
            raise Exception
        ret = func(input_array)
        return ret

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
                grp.create_dataset(path, data=dataset)
            series.append({"path": path})
        return grp

    def nearest(self, image: ArrayLike) -> ArrayLike:
        """
        Downsample using :func:`skimage.transform.resize`.
        """
        if isinstance(image, da.Array):
            _resize = lambda image, out_shape, **kwargs: dask_resize(image, out_shape)
        else:
            _resize = skimage_resize

        # only down-sample in X and Y dimensions for now...
        out_shape = list(image.shape)
        out_shape[-1] = np.ceil(float(image.shape[-1]) / self.downscale)
        out_shape[-2] = np.ceil(float(image.shape[-2]) / self.downscale)

        dtype = image.dtype
        image = _resize(
            image.astype(float), out_shape, order=1, mode="reflect", anti_aliasing=False
        )
        return image.astype(dtype)

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
