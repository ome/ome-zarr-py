import pytest

from ome_zarr.data import create_zarr
from ome_zarr.io import parse_url
from ome_zarr.reader import Layer


class TestLayer:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = tmpdir.mkdir("data")
        create_zarr(str(self.path))

    def test_image(self):
        layer = Layer(parse_url(str(self.path)), list())
        assert layer.data
        assert layer.metadata

    def test_labels(self):
        filename = str(self.path.join("labels"))
        layer = Layer(parse_url(filename), list())
        assert not layer.data
        assert not layer.metadata

    def test_label(self):
        filename = str(self.path.join("labels", "coins"))
        layer = Layer(parse_url(filename), list())
        assert layer.data
        assert layer.metadata
