import pathlib

import numpy as np
import pytest
import zarr

from ome_zarr.io import parse_url
from ome_zarr.reader import Multiscales, Reader
from ome_zarr.writer import write_image


class TestUpgrade:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = pathlib.Path(tmpdir.mkdir("data"))
        self.store = parse_url(self.path, mode="w").store
        self.root = zarr.group(store=self.store)
        self.group = self.root.create_group("test")

    def create_data(self, shape, dtype=np.uint8, mean_val=10):
        rng = np.random.default_rng(0)
        return rng.poisson(mean_val, size=shape).astype(dtype)

    def assert_data(self, path, shape):
        reader = Reader(parse_url(path))
        node = list(reader())[0]
        assert Multiscales.matches(node.zarr)
        assert node.data[0].shape == shape
        assert np.max(node.data[0]) > 0

    def test_existing(self, request):
        shape = (1, 2, 1, 16, 16)
        self.assert_data(f"{request.fspath.dirname}/data/v1", shape)
        self.assert_data(f"{request.fspath.dirname}/data/v2", shape)

    @pytest.mark.parametrize(
        "from_version,to_version",
        (
            pytest.param("0.1", "0.1"),
            pytest.param("0.1", "0.2"),
            pytest.param("0.2", "0.2"),
        ),
    )
    def test_upgrade(self, from_version, to_version):
        shape = (1, 1, 1, 4, 4)
        data = self.create_data(shape)
        write_image(image=data, group=self.group, chunks=(128, 128), scaler=None)
        self.assert_data(f"{self.path}/test", shape)
