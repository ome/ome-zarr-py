import pathlib

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

    def check_downscaled(self, downscaled, shape, scale_factor=2):
        expected_shape = shape
        for data in downscaled:
            assert data.shape == expected_shape
            assert data.dtype == downscaled[0].dtype
            expected_shape = expected_shape[:-2] + tuple(
                sh // scale_factor for sh in expected_shape[-2:]
            )

    def test_nearest(self, shape):
        data = self.create_data(shape)
        scaler = Scaler()
        downscaled = scaler.nearest(data)
        self.check_downscaled(downscaled, shape)

    def test_nearest_via_method(self, shape):
        data = self.create_data(shape)

        scaler = Scaler()
        expected_downscaled = scaler.nearest(data)

        scaler.method = "nearest"
        downscaled = scaler.func(data)
        self.check_downscaled(downscaled, shape)

        assert (
            np.sum(
                [
                    not np.array_equal(downscaled[i], expected_downscaled[i])
                    for i in range(len(downscaled))
                ]
            )
            == 0
        )

    # this fails because of wrong channel dimension; need to fix in follow-up PR
    @pytest.mark.xfail
    def test_gaussian(self, shape):
        data = self.create_data(shape)
        scaler = Scaler()
        downscaled = scaler.gaussian(data)
        self.check_downscaled(downscaled, shape)

    # this fails because of wrong channel dimension; need to fix in follow-up PR
    @pytest.mark.xfail
    def test_laplacian(self, shape):
        data = self.create_data(shape)
        scaler = Scaler()
        downscaled = scaler.laplacian(data)
        self.check_downscaled(downscaled, shape)

    def test_local_mean(self, shape):
        data = self.create_data(shape)
        scaler = Scaler()
        downscaled = scaler.local_mean(data)
        self.check_downscaled(downscaled, shape)

    def test_local_mean_via_method(self, shape):
        data = self.create_data(shape)

        scaler = Scaler()
        expected_downscaled = scaler.local_mean(data)

        scaler.method = "local_mean"
        downscaled = scaler.func(data)
        self.check_downscaled(downscaled, shape)

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
        self.check_downscaled(downscaled, shape)

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

    @pytest.mark.parametrize("method", ["nearest", "resize", "laplacian", "local_mean", "zoom"])
    @pytest.mark.parametrize("n_levels", [1, 2, 3, 4])
    def test_build_pyramid(self, shape, method, n_levels):
        from ome_zarr.scale import _build_pyramid

        data = self.create_data(shape)

        if len(data.shape) == 5:
            dims = ("t", "c", "z", "y", "x")
        elif len(data.shape) == 3:
            dims = ("z", "y", "x")
        elif len(data.shape) == 2:
            dims = ("y", "x")

        scale_factors = [2**i for i in range(1, n_levels)]
        pyramid = _build_pyramid(
            image=data,
            scale_factors=scale_factors,
            method=method,
            dims=dims,
        )

        assert len(pyramid) == n_levels  # original + (n_levels - 1) downscaled
        assert pyramid[0].shape == data.shape

        # Make sure channel and time dimensions are preserved
        for level in pyramid:
            for idx, d in enumerate(dims):
                if d in ("t", "c"):
                    assert level.shape[idx] == data.shape[idx]

        for idx, level in enumerate(pyramid[1:], start=1):
            previous_shape = pyramid[idx - 1].shape
            current_shape = level.shape

            if idx == 1:
                relative_scale = scale_factors[0]
            else:
                relative_scale = scale_factors[idx - 1] // scale_factors[idx - 2]
            assert current_shape[-2] == previous_shape[-2] // relative_scale

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
