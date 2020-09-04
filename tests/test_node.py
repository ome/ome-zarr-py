import pytest

from ome_zarr.data import create_zarr
from ome_zarr.io import parse_url
from ome_zarr.reader import Node


class TestNode:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = tmpdir.mkdir("data")
        create_zarr(str(self.path))

    def test_image(self):
        node = Node(parse_url(str(self.path)), list())
        assert node.data
        assert node.metadata

    def test_labels(self):
        filename = str(self.path.join("labels"))
        node = Node(parse_url(filename), list())
        assert not node.data
        assert not node.metadata

    def test_label(self):
        filename = str(self.path.join("labels", "coins"))
        node = Node(parse_url(filename), list())
        assert node.data
        assert node.metadata
