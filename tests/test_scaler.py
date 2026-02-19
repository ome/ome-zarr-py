import pathlib
import tempfile

import dask.array as da
import numpy as np
import pytest
import zarr

from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image


class TestScaler:
    @pytest.fixture(
        params=(
            (1, 2, 1, 256, 256),
            (3, 512, 512),
            (256, 256),
        ),
        ids=["5D", "3D", "2D"],
    )
    def shape(self, request):
        return request.param

    def create_data(self, shape, dtype=np.uint8, mean_val=10):
        rng = np.random.default_rng(0)
        return rng.poisson(mean_val, size=shape).astype(dtype)

    def check_downscaled(self, downscaled, data, scale_factor=2):
        expected_shape = data.shape
        for level in downscaled:
            assert level.dtype == data.dtype
            if scale_factor is not None:
                assert level.shape == expected_shape
                expected_shape = expected_shape[:-2] + tuple(
                    sh // scale_factor for sh in expected_shape[-2:]
                )

    def test_nearest(self, shape):
        data = self.create_data(shape)
        scaler = Scaler()
        downscaled = scaler.nearest(data)
        self.check_downscaled(downscaled, data)

    def test_nearest_via_method(self, shape):
        data = self.create_data(shape)

        scaler = Scaler()
        expected_downscaled = scaler.nearest(data)

        scaler.method = "nearest"
        downscaled = scaler.func(data)
        self.check_downscaled(downscaled, data)

        assert (
            np.sum(
                [
                    not np.array_equal(downscaled[i], expected_downscaled[i])
                    for i in range(len(downscaled))
                ]
            )
            == 0
        )

    # NB: gaussian downscales ALL dimensions, not just YX
    # so we SKIP the check on shape
    def test_gaussian(self, shape):
        data = self.create_data(shape)
        scaler = Scaler()
        downscaled = scaler.gaussian(data)
        self.check_downscaled(downscaled, data, scale_factor=None)

    # NB: laplacian downscales ALL dimensions, not just YX
    # so we SKIP the check on shape
    def test_laplacian(self, shape):
        data = self.create_data(shape)
        scaler = Scaler()
        downscaled = scaler.laplacian(data)
        self.check_downscaled(downscaled, data, scale_factor=None)

    def test_local_mean(self, shape):
        data = self.create_data(shape)
        scaler = Scaler()
        downscaled = scaler.local_mean(data)
        self.check_downscaled(downscaled, data)

    def test_local_mean_via_method(self, shape):
        data = self.create_data(shape)

        scaler = Scaler()
        expected_downscaled = scaler.local_mean(data)

        scaler.method = "local_mean"
        downscaled = scaler.func(data)
        self.check_downscaled(downscaled, data)

        assert (
            np.sum(
                [
                    not np.array_equal(downscaled[i], expected_downscaled[i])
                    for i in range(len(downscaled))
                ]
            )
            == 0
        )

    @pytest.mark.skip(reason="This test does not terminate")
    def test_zoom(self, shape):
        data = self.create_data(shape)
        scaler = Scaler()
        downscaled = scaler.zoom(data)
        self.check_downscaled(downscaled, data)

    def test_scale_dask(self, shape):
        data = self.create_data(shape)
        # chunk size gives odd-shaped chunks at the edges
        # tests https://github.com/ome/ome-zarr-py/pull/244
        chunk_size = [100, 100]
        chunk_2d = (*(1,) * (data.ndim - 2), *chunk_size)

        data_delayed = da.from_array(data, chunks=chunk_2d)

        scaler = Scaler()
        resized_data = scaler.resize_image(data)
        resized_dask = scaler.resize_image(data_delayed)

        assert np.array_equal(resized_data, resized_dask)

    def test_scale_dask_via_method(self, shape):
        data = self.create_data(shape)

        chunk_size = [100, 100]
        chunk_2d = (*(1,) * (data.ndim - 2), *chunk_size)
        data_delayed = da.from_array(data, chunks=chunk_2d)

        scaler = Scaler()
        expected_downscaled = scaler.resize_image(data)

        scaler.method = "resize_image"
        assert np.array_equal(expected_downscaled, scaler.func(data))
        assert np.array_equal(expected_downscaled, scaler.func(data_delayed))

    def test_big_dask_pyramid(self, tmpdir):
        # from https://github.com/ome/omero-cli-zarr/pull/134
        shape = (6675, 9560)
        data = self.create_data(shape)
        data_delayed = da.from_array(data, chunks=(1000, 1000))
        print("data_delayed", data_delayed)
        scaler = Scaler()
        level_1 = scaler.resize_image(data_delayed)
        print("level_1", level_1)
        # to zarr invokes compute
        data_dir = tmpdir.mkdir("test_big_dask_pyramid")
        da.to_zarr(level_1, str(data_dir))

    @pytest.mark.parametrize("method", ["nearest", "resize", "local_mean", "zoom"])
    @pytest.mark.parametrize(
        "shape, scale_factors, dims",
        [
            (
                (1, 2, 1, 256, 256),
                [
                    {"t": 1, "c": 1, "z": 1, "y": 2, "x": 2},
                    {"t": 1, "c": 1, "z": 1, "y": 4, "x": 4},
                    {"t": 1, "c": 1, "z": 1, "y": 8, "x": 8},
                ],
                ["t", "c", "z", "y", "x"],
            ),
            (
                #  check that missing t and c dimensions should be coerced to 1 and preserved
                (1, 2, 1, 256, 256),
                [
                    {"z": 1, "y": 2, "x": 2},
                    {"z": 1, "y": 4, "x": 4},
                    {"z": 1, "y": 8, "x": 8},
                ],
                ["t", "c", "z", "y", "x"],
            ),
            (
                # check that z is properly downsampled when requested
                (128, 128, 128),
                [
                    {"z": 2, "y": 2, "x": 2},
                    {"z": 4, "y": 4, "x": 4},
                    {"z": 8, "y": 8, "x": 8},
                ],
                ["z", "y", "x"],
            ),
            (
                (256, 256),
                [
                    {"y": 2, "x": 2},
                    {"y": 4, "x": 4},
                    {"y": 8, "x": 8},
                ],
                ["y", "x"],
            ),
        ],
    )
    def test_build_pyramid(self, shape, scale_factors, dims, method):

        data = self.create_data(shape)

        # write the pyramid to zarr to make sure it works with dask arrays
        with tempfile.TemporaryDirectory() as tmpdir:
            write_image(
                data,
                group=zarr.open_group(tmpdir, mode="w"),
                scale_factors=scale_factors,
                axes=dims,
                method=method,
            )

            # read back the pyramid to check it was written correctly
            pyramid = [
                da.from_array(zarr.open_group(tmpdir, mode="r")[f"{i}"])
                for i in range(len(scale_factors) + 1)
            ]

            assert (
                len(pyramid) == len(scale_factors) + 1
            )  # original + (n_levels - 1) downscaled
            assert pyramid[0].shape == data.shape

            if isinstance(scale_factors[0], int):
                scale_factors = [
                    {d: scale_factors[i] if d in ("y", "x") else 1 for d in dims}
                    for i in range(1, len(scale_factors) + 1)
                ]

            # check if factors for z are different from 1 across levels
            # to determine if z should have been downsampled
            z_factors = [sf.get("z") for sf in scale_factors]
            if all(zf == 1 for zf in z_factors):
                downsample_z = False
            else:
                downsample_z = True

            # Make sure channel and time dimensions are preserved
            for level in pyramid:
                for idx, d in enumerate(dims):
                    if d in ("t", "c"):
                        assert level.shape[idx] == data.shape[idx]

                    # make sure z is not downsampled by default unless specifically requested
                    if "z" in dims and not downsample_z:
                        assert (
                            level.shape[dims.index("z")] == data.shape[dims.index("z")]
                        )

            for idx, level in enumerate(pyramid[1:], start=1):
                previous_shape = pyramid[idx - 1].shape
                current_shape = level.shape

                if idx == 1:
                    previous_scale_factor = dict.fromkeys(dims, 1)
                    current_scale_factor = scale_factors[0]
                else:
                    previous_scale_factor = scale_factors[idx - 2]
                    current_scale_factor = scale_factors[idx - 1]

                relative_factor = {
                    d: current_scale_factor[d] / previous_scale_factor[d] for d in dims
                }

                # Check all spatial dimensions are scaled correctly
                for dim_idx, dim_name in enumerate(dims):
                    if dim_name in ("y", "x", "z"):
                        expected_dim_size = int(
                            np.ceil(previous_shape[dim_idx] / relative_factor[dim_name])
                        )
                        assert current_shape[dim_idx] == expected_dim_size

    @pytest.mark.parametrize("method", ["gaussian", "laplacian"])
    def test_pyramid_args(self, shape, tmpdir, method):
        path = pathlib.Path(tmpdir.mkdir("data"))
        group = zarr.open_group(path, mode="w")

        data = self.create_data(shape)

        scaler = Scaler(
            downscale=2,
            method=method,
            max_layer=2,
        )

        axes = "tczyx"[-len(shape) :]
        write_image(
            image=data,
            group=group,
            scaler=scaler,
            axes=axes,
        )


if __name__ == "__main__":
    pytest.main([__file__])
