import numpy as np
import pytest

from ome_zarr.reader import Reader
from ome_zarr.writer import write_image
from ome_zarr.io import parse_url


class TestWriter:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = tmpdir.mkdir("data")

    def create_data(self, shape, dtype, mean_val=10):
        rng = np.random.default_rng(0)
        return rng.poisson(mean_val, size=shape).astype(dtype)

    def test_writer(self):

        data = self.create_data((1, 2, 1, 256, 256), np.uint8)
        write_image(self.path, data, name="test")
        reader = Reader(parse_url(self.path))
        assert list(reader())
