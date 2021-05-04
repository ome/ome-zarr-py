"""Module for downsampling numpy arrays via various methods.

See the :class:`~ome_zarr.scale.Scaler` class for details.
"""
import inspect
import logging
import os
import shutil
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Tuple

import cv2
import dask.array as da
import numpy as np
import zarr
from scipy.ndimage import zoom
from skimage.transform import (
    downscale_local_mean,
    pyramid_gaussian,
    pyramid_laplacian,
    rescale,
)
from zarr.storage import FSStore

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
    downsample_z: bool = False
    in_place: bool = False
    labeled: bool = False
    max_layer: int = 4
    method: str = "nearest"
    output_directory: Optional[str] = None
    shape: Optional[Tuple[int, int, int]] = None
    dtype: Optional[np.dtype] = None

    @staticmethod
    def methods() -> Iterator[str]:
        """Return the name of all methods which define a downsampling.

        Any of the returned values can be used as the `methods`
        argument to the
        :func:`Scaler constructor <ome_zarr.scale.Scaler._init__>`
        """
        funcs = inspect.getmembers(Scaler, predicate=inspect.isfunction)
        for name, func in funcs:
            if name in (
                "methods",
                "scale",
                "z_scale_pyramid",
                "add_plane_to_pyramid",
                "scale_arrays_3d",
                "scale_array_xy_to_pyramid",
            ):
                continue
            if name.startswith("_"):
                continue
            yield name

    def scale(
        self,
        input_array_or_group: str,
        output_directory: str,
        downsample_z: bool = False,
    ) -> None:
        """
        Perform downsampling to disk.

        If input_array_or_group is a path/to/array (contains .zarray) then this creates
        a pyramid of resolution levels (arrays in the output_directory),
        downsampling X and Y.
        If downsample_z is True, then we subsequently downsample the pyramid in Z.

        If input_array_or_group is a path/to/group (contains .zgroup) and
        downsample_z is True, this creates a new pyramid, in the output_directory,
        downsampling Z only.
        """

        # If input is array, first downsample XY to create pyramid
        pyramid_dir = None
        input_zarray = os.path.join(input_array_or_group, ".zarray")
        if os.path.exists(input_zarray):
            print("downsampling in X and Y to create pyramid...")
            if downsample_z:
                # will delete this once downsample_z is done
                pyramid_dir = "%s_temp" % output_directory
            else:
                pyramid_dir = output_directory

            if self.method == "nearest":
                # Writes each plane to disk in turn
                self.scale_array_xy_to_pyramid(input_array_or_group, pyramid_dir)
            else:
                func = getattr(self, self.method, None)
                if not func:
                    raise Exception

                store = self.__check_store(pyramid_dir)
                base = zarr.open_array(input_array_or_group)
                pyramid = func(base)

                if self.labeled:
                    self.__assert_values(pyramid)

                grp = self.__create_group(store, base, pyramid)

            if self.copy_metadata:
                print(f"copying attribute keys: {list(base.attrs.keys())}")
                grp.attrs.update(base.attrs)

        elif os.path.exists(os.path.join(input_array_or_group, ".zgroup")):
            # if input is a .zgroup
            if not downsample_z:
                raise ValueError(
                    "If input is a pyramid, use" " --downsample_z to downsample"
                )
        else:
            raise ValueError("input is not a zarr array or group")

        if downsample_z:
            print("downsampling pyramid in Z...")
            zscale_input = input_array_or_group
            if pyramid_dir is not None:
                zscale_input = pyramid_dir

            self.z_scale_pyramid(zscale_input, output_directory)

            if pyramid_dir is not None:
                print("Deleting temp ", pyramid_dir)
                shutil.rmtree(pyramid_dir)

    def _open_store(self, name: str) -> FSStore:
        """
        Create an FSStore instance that supports nested storage of chunks.
        """
        return FSStore(
            name,
            auto_mkdir=True,
            key_separator="/",
            normalize_keys=False,
            mode="w",
        )

    def add_plane_to_pyramid(
        self,
        plane: np.ndarray,
        indices: Tuple[int, int, int],
        func: Optional[Callable] = None,
        get_level_name: Optional[Callable] = None,
    ) -> None:
        """
        Adds a 2D numpy plane to each level of a pyramid, at (t, c, z) indices.

        If no downsample function is provided, use cv2.resize() with
        self.downsample factor (default is 2) and interpolation=cv2.INTER_NEAREST

        Usage:
        # pyramid shape can be (t, c, z, y, x) or (t, c, z)
        scaler = Scaler(output_dir, shape, dtype, level_count)
        scaler.add_plane_to_pyramid(plane_2d, (t, c, z), func=scale_2d)
        scaler.add_plane_to_pyramid(plane_2d, (t, c, z), get_level_name=name_func)
        """

        output_directory = self.output_directory
        shape = self.shape
        dtype = self.dtype
        level_count = self.max_layer

        assert output_directory is not None
        assert shape is not None
        assert dtype is not None
        assert level_count is not None

        # pyramid shape could be given by (t, c, z, y, x) or (t, c, z)
        assert len(shape) > 2
        store = self._open_store(output_directory)
        parent = zarr.group(store)

        size_t: int = shape[0]
        size_c: int = shape[1]
        size_z: int = shape[2]

        t, c, z = indices

        if func is None:

            def func(plane_2d: np.ndarray) -> np.ndarray:
                size_x = plane_2d.shape[-1] // self.downscale
                size_y = plane_2d.shape[-2] // self.downscale
                return cv2.resize(
                    plane_2d,
                    dsize=(size_x, size_y),
                    interpolation=cv2.INTER_NEAREST,
                )

        assert func is not None

        for level in range(level_count):
            if level > 0:
                # downsample
                plane = func(plane)
            # size x and y change with each level. z, c, t don't change
            size_y = plane.shape[0]
            size_x = plane.shape[1]
            dataset = parent.require_dataset(
                get_level_name(level) if get_level_name is not None else str(level),
                shape=(size_t, size_c, size_z, size_y, size_x),
                chunks=(1, 1, 1, size_y, size_x),
                dtype=dtype,
            )

            dataset[t, c, z, :, :] = plane

    def scale_array_xy_to_pyramid(
        self, input_array: str, output_directory: str
    ) -> None:
        """
        Scales the array to create a pyramid in output_directory.

        Process the T/C/Z planes one at a time, writing them to the
        pyramid on disk in turn.
        """
        base = da.from_zarr(input_array)
        self.dtype = base.dtype
        self.output_directory = output_directory
        size_t, size_c, size_z, size_y, size_x = base.shape
        self.shape = (size_t, size_c, size_z)

        for t in range(size_t):
            for c in range(size_c):
                for z in range(size_z):
                    plane_2d = base[t, c, z, :, :]
                    plane_2d = plane_2d.compute()
                    self.add_plane_to_pyramid(plane_2d, (t, c, z))

    def scale_arrays_3d(
        self,
        input_group: str,
        output_group: str,
        func3d: Callable[[np.ndarray, str], np.ndarray],
        input_arrays: Optional[List[str]] = None,
        output_arrays: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Process 1 or more arrays within an input_group, applying a func3d transform.

        Output arrays will be created in the output_group, with the same names as the
        input arrays by default.
        """

        if overwrite is False:
            if output_group is None and output_arrays is None:
                # make sure we don't accidentally overwrite
                raise TypeError(
                    """Need to set overwrite=True, or specify
                    output_group or output_arrays"""
                )

        if input_arrays is None:
            input_store = zarr.DirectoryStore(input_group)
            input_grp = zarr.group(input_store)
            input_arrays = list(input_grp.array_keys())

        # write output to new arrays in the same group, or overwrite arrays
        # if same level names
        if output_group is None:
            output_group = input_group

        # create arrays named the same as input arrays, or overwrite arrays
        # if same group
        if output_arrays is None:
            output_arrays = input_arrays

        # output_directory may already exist
        store = self._open_store(output_group)
        grp = zarr.group(store)

        for input_level, output_level in zip(input_arrays, output_arrays):
            if input_group is not None:
                input_path = os.path.join(input_group, input_level)
            else:
                input_path = input_level
            base = da.from_zarr(input_path)

            size_t = base.shape[0]
            size_c = base.shape[1]
            for t in range(size_t):
                for c in range(size_c):
                    data_3d = base[t, c, :, :, :]
                    resized = func3d(data_3d, input_level)
                    size_z, size_y, size_x = resized.shape
                    dataset = grp.require_dataset(
                        output_level,
                        shape=(size_t, size_c, size_z, size_y, size_x),
                        chunks=(1, 1, 1, size_y, size_x),
                        dtype=resized.dtype,
                    )
                    dataset[t, c, :, :, :] = resized

    def z_scale_pyramid(self, input_group: str, output_group: str) -> None:
        """Downsample a SINGLE 5D array in Z and write to output_group"""

        # Copy level '0' and '.zattrs' to new dir
        if os.path.exists(os.path.join(input_group, "0")) and not os.path.exists(
            os.path.join(output_group, "0")
        ):
            shutil.copytree(
                os.path.join(input_group, "0"), os.path.join(output_group, "0")
            )
        if os.path.exists(os.path.join(input_group, ".zattrs")) and not os.path.exists(
            os.path.join(output_group, ".zattrs")
        ):
            shutil.copyfile(
                os.path.join(input_group, ".zattrs"),
                os.path.join(output_group, ".zattrs"),
            )

        def doscale(array_3d: np.ndarray, array_name: str) -> np.ndarray:
            # Assume input pyramid levels are named '0', '1', '2' etc.
            input_level = int(array_name)
            factor = (1 / self.downscale) ** input_level
            rescaled = rescale(array_3d, (factor, 1, 1), preserve_range=True)
            # preserve input dtype
            return rescaled.astype(array_3d.dtype)

        input_store = zarr.DirectoryStore(input_group)
        input_grp = zarr.group(input_store)
        # Don't process level 0
        input_arrays = [arr for arr in input_grp.array_keys() if arr != "0"]
        output_arrays = None
        # If creating arrays in the *same* group as input arrays, rename:
        if input_group == output_group:
            output_arrays = ["z%s" % arr for arr in input_arrays]

        self.scale_arrays_3d(
            input_group,
            output_group,
            doscale,
            input_arrays=input_arrays,
            output_arrays=output_arrays,
        )

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
                    f"a subset of {len(expected)} values"
                )

    def __create_group(
        self, store: MutableMapping, base: np.ndarray, pyramid: List[np.ndarray]
    ) -> zarr.hierarchy.Group:
        """Create group and datasets."""
        grp = zarr.group(store)
        series = []
        for i, dataset in enumerate(pyramid):
            path = "%s" % i
            grp.create_dataset(path, data=pyramid[i])
            series.append({"path": path})
        return grp

    def nearest(self, base: np.ndarray) -> List[np.ndarray]:
        """
        Downsample using :func:`cv2.resize`.

        The :const:`cvs2.INTER_NEAREST` interpolation method is used.
        """
        return self._by_plane(base, self.__nearest)

    def __nearest(self, plane: np.ndarray, sizeY: int, sizeX: int) -> np.ndarray:
        """Apply the 2-dimensional transformation."""
        return cv2.resize(
            plane,
            dsize=(sizeX // self.downscale, sizeY // self.downscale),
            interpolation=cv2.INTER_NEAREST,
        )

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
        """Downsample using :func:`scipy.ndimage.zoom`."""
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
        self,
        base: np.ndarray,
        func: Callable[[np.ndarray, int, int], np.ndarray],
    ) -> np.ndarray:
        """Loop over 3 of the 5 dimensions and apply the func transform."""
        assert 5 == len(base.shape)

        rv = [base]
        for i in range(self.max_layer):
            fiveD = rv[-1]
            # FIXME: fix hard-coding of dimensions
            T, C, Z, Y, X = fiveD.shape

            smaller = None
            for t in range(T):
                for c in range(C):
                    z_stack = []
                    for z in range(Z):
                        orig = fiveD[t][c][z][:]
                        p = func(orig, Y, X)
                        z_stack.append(p)
                    temp_arr = np.stack(z_stack)

                    if smaller is None:
                        smaller = np.zeros(
                            (
                                T,
                                C,
                                temp_arr.shape[0],
                                temp_arr.shape[1],
                                temp_arr.shape[2],
                            ),
                            dtype=base.dtype,
                        )

                    smaller[t][c] = temp_arr

            rv.append(smaller)
        return rv
