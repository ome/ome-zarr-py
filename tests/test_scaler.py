import numpy as np
import pytest
from ome_zarr.scale import Scaler


class TestScaler:
    @pytest.fixture(params=((1, 2, 1, 256, 256),))
    def shape(self, request):
        return request.param

    def create_data(self, shape, dtype=np.uint8, mean_val=10):
        rng = np.random.default_rng(0)
        return rng.poisson(mean_val, size=shape).astype(dtype)

    def check_downscaled(self, downscaled, shape, scale_factor=2):
        expected_shape = shape
        for data in downscaled:
            assert data.shape == expected_shape
            expected_shape = expected_shape[:-2] + tuple(sh // scale_factor for sh in expected_shape[-2:])

    def test_nearest(self, shape):
        data = self.create_data(shape)
        scaler = Scaler()
        downscaled = scaler.nearest(data)
        self.check_downscaled(downscaled, shape)

    # this fails because of wrong channel dimension; mark expected failure?
    def test_gaussian(self, shape):
        data = self.create_data(shape)
        scaler = Scaler()
        downscaled = scaler.gaussian(data)
        self.check_downscaled(downscaled, shape)

    # this fails because of wrong channel dimension; mark expected failure?
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

    @pytest.mark.skip(reason="This test does not terminate")
    def test_zoom(self, shape):
        data = self.create_data(shape)
        scaler = Scaler()
        downscaled = scaler.zoom(data)
        self.check_downscaled(downscaled, shape)
