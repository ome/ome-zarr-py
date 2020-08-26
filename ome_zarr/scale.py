import inspect
import logging
import os
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Callable, Iterator, List

import cv2
import numpy as np
import zarr
from scipy.ndimage import zoom
from skimage.transform import downscale_local_mean, pyramid_gaussian, pyramid_laplacian

LOGGER = logging.getLogger("ome_zarr.scale")


@dataclass
class Scaler:

    copy_metadata: bool = False
    downscale: int = 2
    in_place: bool = False
    labeled: bool = False
    max_layer: int = 4
    method: str = "nearest"

    @staticmethod
    def methods() -> Iterator[str]:
        funcs = inspect.getmembers(Scaler, predicate=inspect.isfunction)
        for name, func in funcs:
            if name in ("methods", "scale"):
                continue
            if name.startswith("_"):
                continue
            yield name

    def scale(self, input_array: str, output_directory: str) -> None:

        func = getattr(self, self.method, None)
        if not func:
            raise Exception

        store = self._check_store(output_directory)
        base = zarr.open_array(input_array)
        pyramid = func(base)

        if self.labeled:
            self._assert_values(pyramid)

        grp = self._create_group(store, base, pyramid)

        if self.copy_metadata:
            print(f"copying attribute keys: {list(base.attrs.keys())}")
            grp.attrs.update(base.attrs)

    def _check_store(self, output_directory: str) -> MutableMapping:
        assert not os.path.exists(output_directory)
        return zarr.DirectoryStore(output_directory)

    def _assert_values(self, pyramid: List[np.ndarray]) -> None:
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

    def _create_group(
        self, store: MutableMapping, base: np.ndarray, pyramid: List[np.ndarray]
    ) -> zarr.hierarchy.Group:
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

    #
    # Scaling methods
    #

    def nearest(self, base: np.ndarray) -> List[np.ndarray]:
        def func(plane: np.ndarray, sizeY: int, sizeX: int) -> np.ndarray:
            return cv2.resize(
                plane,
                dsize=(sizeY // self.downscale, sizeX // self.downscale),
                interpolation=cv2.INTER_NEAREST,
            )

        return self._by_plane(base, func)

    def gaussian(self, base: np.ndarray) -> List[np.ndarray]:
        return list(
            pyramid_gaussian(
                base,
                downscale=self.downscale,
                max_layer=self.max_layer,
                multichannel=False,
            )
        )

    def laplacian(self, base: np.ndarray) -> List[np.ndarray]:
        return list(
            pyramid_laplacian(
                base,
                downscale=self.downscale,
                max_layer=self.max_layer,
                multichannel=False,
            )
        )

    def local_mean(self, base: np.ndarray) -> List[np.ndarray]:
        # FIXME: fix hard-coding
        rv = [base]
        for i in range(self.max_layer):
            rv.append(
                downscale_local_mean(
                    rv[-1], factors=(1, 1, 1, self.downscale, self.downscale)
                )
            )
        return rv

    def zoom(self, base: np.ndarray) -> List[np.ndarray]:
        rv = [base]
        print(base.shape)
        for i in range(self.max_layer):
            print(i, self.downscale)
            rv.append(zoom(base, self.downscale ** i))
            print(rv[-1].shape)
        return list(reversed(rv))

    #
    # Helpers
    #

    def _by_plane(
        self, base: np.ndarray, func: Callable[[np.ndarray, int, int], np.ndarray],
    ) -> np.ndarray:

        assert 5 == len(base.shape)

        rv = [base]
        for i in range(self.max_layer):
            fiveD = rv[-1]
            # FIXME: fix hard-coding of dimensions
            T, C, Z, Y, X = fiveD.shape

            smaller = None
            for t in range(T):
                for c in range(C):
                    for z in range(Z):
                        out = func(fiveD[t][c][z][:], Y, X)
                        if smaller is None:
                            smaller = np.zeros((T, C, Z, out.shape[0], out.shape[1]))
                        smaller[t][c][z] = out
            rv.append(smaller)
        return rv
