import pathlib

import numpy as np
import pytest
import zarr

from ome_zarr.format import CurrentFormat
from ome_zarr.io import parse_url
from ome_zarr.reader import Multiscales, Reader


class TestUpgrade:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = pathlib.Path(tmpdir.mkdir("data"))

    def create_data(self, shape, fmt=CurrentFormat(), dtype=np.uint8, mean_val=128):
        self.store = parse_url(self.path, mode="w", fmt=fmt).store
        self.root = zarr.group(store=self.store)
        self.group = self.root.create_group("test")
        rng = np.random.default_rng(0)
        return rng.poisson(mean_val, size=shape).astype(dtype)

    def assert_data(self, path, shape, fmt, mode="r"):
        loc = parse_url(path, mode=mode, fmt=fmt)
        assert loc
        reader = Reader(loc)
        node = next(iter(reader()))
        assert Multiscales.matches(node.zarr)
        assert node.data[0].shape == shape
        assert np.max(node.data[0]) > 0
        assert loc.fmt == fmt

    def test_requested_no_upgrade(self):
        print("warnings")

    def test_automatic_upgrade(self):
        print("info or debug?")

    def test_cli_upgrade(self):
        print("dry-run?")
