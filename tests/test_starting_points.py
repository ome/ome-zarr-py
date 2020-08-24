import tempfile
from typing import Set, Type

from ome_zarr.data import create_zarr
from ome_zarr.io import parse_url
from ome_zarr.reader import OMERO, Label, Labels, Layer, Multiscales, Spec


class TestStartingPoints:
    """
    Creates a small but complete OME-Zarr file and tests that
    readers will detect the correct type when starting at all
    the various levels.
    """

    @classmethod
    def setup_class(cls):
        """
        """
        cls.path = tempfile.TemporaryDirectory(suffix=".zarr").name
        create_zarr(cls.path)

    def matches(self, layer: Layer, expected: Set[Type[Spec]]):
        found: Set[Type[Spec]] = set()
        for spec in layer.specs:
            found.add(type(spec))
        assert expected == found

    def test_top_level(self):
        zarr = parse_url(self.path)
        assert zarr is not None
        layer = Layer(zarr)
        self.matches(layer, set([Multiscales, OMERO]))

    def test_labels(self):
        zarr = parse_url(self.path + "/labels")
        assert zarr is not None
        layer = Layer(zarr)
        self.matches(layer, set([Labels]))

    def test_label(self):
        zarr = parse_url(self.path + "/labels/coins")
        assert zarr is not None
        layer = Layer(zarr)
        self.matches(layer, set([Label, Multiscales]))
