import numpy as np
import pytest
import zarr

from ome_zarr.io import parse_url
from ome_zarr.reader import Multiscales, Reader
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image


class TestWriter:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = tmpdir.mkdir("data")
        self.store = zarr.DirectoryStore(self.path)
        self.root = zarr.group(store=self.store)
        self.group = self.root.create_group("test")

    def create_data(self, shape, dtype=np.uint8, mean_val=10):
        rng = np.random.default_rng(0)
        return rng.poisson(mean_val, size=shape).astype(dtype)

    @pytest.fixture(params=((1, 2, 1, 256, 256),))
    def shape(self, request):
        return request.param

    @pytest.fixture(params=[True, False], ids=["flat", "list"])
    def data(self, shape, request):
        rv = self.create_data(shape)
        if request.param:
            return rv
        else:
            return [rv]

    @pytest.fixture(params=[True, False], ids=["scale", "noop"])
    def scaler(self, request):
        if request.param:
            return Scaler()
        else:
            return None

    def test_writer(self, shape, data, scaler):

        write_image(image=data, group=self.group, chunks=(128, 128), scaler=scaler)

        # Verify
        reader = Reader(parse_url(f"{self.path}/test"))
        node = list(reader())[0]
        assert Multiscales.matches(node.zarr)
        assert node.data[0].shape == shape
        assert node.data[0].chunks == ((1,), (2,), (1,), (128, 128), (128, 128))
