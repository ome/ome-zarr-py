import pytest

from ome_zarr.data import create_zarr
from ome_zarr.io import parse_url
from ome_zarr.reader import OMERO, Label, Labels, Multiscales, Node, Spec


class TestStartingPoints:
    """Creates a small but complete OME-Zarr file and tests that readers will detect the
    correct type when starting at all the various levels."""

    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = tmpdir.mkdir("data")
        create_zarr(str(self.path))

    def matches(self, node: Node, expected: list[type[Spec]]):
        found: list[type[Spec]] = [type(spec) for spec in node.specs]

        expected_names = sorted(x.__name__ for x in expected)
        found_names = sorted(x.__name__ for x in found)
        assert expected_names == found_names

    def get_spec(self, node: Node, spec_type: type[Spec]):
        for spec in node.specs:
            if isinstance(spec, spec_type):
                return spec
        assert False, f"no {spec_type} found"

    def test_top_level(self):
        zarr = parse_url(str(self.path))
        assert zarr is not None
        node = Node(zarr, list())
        self.matches(node, {Multiscales, OMERO})
        multiscales = self.get_spec(node, Multiscales)
        assert multiscales.lookup("multiscales", [])

    def test_labels(self):
        zarr = parse_url(str(self.path + "/labels"))
        assert zarr is not None
        node = Node(zarr, list())
        self.matches(node, {Labels})

    def test_label(self):
        zarr = parse_url(str(self.path + "/labels/coins"))
        assert zarr is not None
        node = Node(zarr, list())
        self.matches(node, {Label, Multiscales})
        multiscales = self.get_spec(node, Multiscales)
        assert multiscales.lookup("multiscales", [])
