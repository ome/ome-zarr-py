import pathlib

import numpy as np
import pytest
import zarr

from ome_zarr.format import FormatV01, FormatV02, FormatV03
from ome_zarr.io import parse_url
from ome_zarr.reader import Multiscales, Reader
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image


class TestWriter:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = pathlib.Path(tmpdir.mkdir("data"))
        self.store = parse_url(self.path, mode="w").store
        self.root = zarr.group(store=self.store)
        self.group = self.root.create_group("test")

    def create_data(self, shape, dtype=np.uint8, mean_val=10):
        rng = np.random.default_rng(0)
        return rng.poisson(mean_val, size=shape).astype(dtype)

    @pytest.fixture(params=((1, 2, 1, 256, 256),))
    def shape(self, request):
        return request.param

    @pytest.fixture(params=[True, False], ids=["scale", "noop"])
    def scaler(self, request):
        if request.param:
            return Scaler()
        else:
            return None

    @pytest.fixture(params=[FormatV01, FormatV02, FormatV03], ids=["v0.1", "v0.2", "v0.3"])
    def format_version(self, request):
        return request.param

    def test_writer(self, shape, scaler, format_version):

        data = self.create_data(shape)
        write_image(image=data, group=self.group, chunks=(128, 128), scaler=scaler, fmt=format_version())

        # Verify
        reader = Reader(parse_url(f"{self.path}/test"))
        node = list(reader())[0]
        assert Multiscales.matches(node.zarr)
        assert node.data[0].shape == shape
        assert node.data[0].chunks == ((1,), (2,), (1,), (128, 128), (128, 128))
        assert np.allclose(data, node.data[0][...].compute())
