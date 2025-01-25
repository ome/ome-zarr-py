import pathlib

import numpy as np
import pytest
import zarr

from ome_zarr.format import CurrentFormat
from ome_zarr.format import FormatV01 as V01
from ome_zarr.format import FormatV02 as V02
from ome_zarr.format import FormatV03 as V03
from ome_zarr.io import parse_url
from ome_zarr.reader import Multiscales, Reader
from ome_zarr.writer import write_image


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

    @pytest.mark.parametrize(
        "path, version",
        (("v1", V01()), ("v2", V02())),
    )
    def test_pre_created(self, request, path, version):
        shape = (1, 2, 1, 16, 16)
        self.assert_data(f"{request.fspath.dirname}/data/{path}", shape, version)

    @pytest.mark.parametrize(
        "version",
        (
            pytest.param(V01(), id="V01"),
            pytest.param(V02(), id="V02"),
            pytest.param(V03(), id="V03"),
        ),
    )
    def test_newly_created(self, version):
        shape = (1, 1, 1, 8, 8)
        data = self.create_data(shape, version)
        axes = None
        if version not in ("0.1", "0.2"):
            axes = "tczyx"
        write_image(image=data, group=self.group, scaler=None, fmt=version, axes=axes)
        self.assert_data(f"{self.path}/test", shape=shape, fmt=version)

    def test_requested_no_upgrade(self):
        print("warnings")

    def test_automatic_upgrade(self):
        print("info or debug?")

    def test_cli_upgrade(self):
        print("dry-run?")
