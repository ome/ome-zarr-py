import numpy as np
import pytest

from ome_zarr.bioformats2raw import bioformats2raw
from ome_zarr.io import parse_url
from ome_zarr.reader import Multiscales, Node, Reader

TEST_DATA = (
    ("fake-series-2.zarr", False),
    ("plate-rows-2.zarr", True),
)

class TestBf2raw:

    def assert_data(self, path, shape, plate, mode="r"):
        loc = parse_url(path, mode=mode)
        assert loc, f"no zarr found at {path}"
        reader = Reader(loc)
        nodes = list(reader())

        assert any([
            bioformats2raw.matches(node.zarr) for node in nodes
        ]), "plugin not detected"

        WHY WITH PLATE

        for node in nodes:
            if bioformats2raw.matches(node.zarr):
                assert "series" in node.metadata, node.metadata
            elif Multiscales.matches(node.zarr):
                assert node.data[0].shape == shape
                assert np.max(node.data[0]) > 0
            elif "OME" in str(node.zarr):
                pass  # Doesn't get parsed directly
            else:
                raise Exception(node)

    @pytest.mark.parametrize("path,plate", TEST_DATA)
    def test_read_static_data(self, request, path, plate):
        shape = (1, 1, 1, 512, 512)
        self.assert_data(f"{request.fspath.dirname}/data/bf2raw/{path}", shape, plate)

    @pytest.mark.parametrize("path,plate", TEST_DATA)
    def test_node_static_data(self, request, path, plate):
        zarr = parse_url(f"{request.fspath.dirname}/data/bf2raw/{path}")
        import pdb; pdb.set_trace()
        node = Node(zarr, [])
        print(node.specs)
