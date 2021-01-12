import numpy as np
import pytest
import zarr

from ome_zarr.io import parse_url
from ome_zarr.reader import OMERO, Reader
from ome_zarr.writer import write_image


class TestWriter:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = tmpdir.mkdir("data")

    def create_data(self, shape, dtype, mean_val=10):
        rng = np.random.default_rng(0)
        return rng.poisson(mean_val, size=shape).astype(dtype)

    def test_writer(self):

        shape = (1, 2, 1, 256, 256)
        data = self.create_data(shape, np.uint8)
        store = zarr.DirectoryStore(self.path)
        root = zarr.group(store=store)
        grp = root.create_group("test")
        write_image(image=data, group=grp, chunks=(128, 128))
        reader = Reader(parse_url(f"{self.path}/test"))
        node = list(reader())[0]
        assert OMERO.matches(node.zarr)
        assert node.data[0].shape == shape
        assert node.data[0].chunks == ((1,), (2,), (1,), (128, 128), (128, 128))
