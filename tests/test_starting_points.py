from typing import List, Type

import pytest

from ome_zarr.data import create_zarr
from ome_zarr.io import parse_url
from ome_zarr.reader import OMERO, Label, Labels, Layer, Multiscales, Spec


class TestStartingPoints:
    """
    Creates a small but complete OME-Zarr file and tests that
    readers will detect the correct type when starting at all
    the various levels.
    """

    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = tmpdir.mkdir("data")
        create_zarr(str(self.path))

    def matches(self, layer: Layer, expected: List[Type[Spec]]):
        found: List[Type[Spec]] = list()
        for spec in layer.specs:
            found.append(type(spec))

        expected_names = sorted([x.__class__.__name__ for x in expected])
        found_names = sorted([x.__class__.__name__ for x in found])
        assert expected_names == found_names

    def get_spec(self, layer: Layer, spec_type: Type[Spec]):
        for spec in layer.specs:
            if isinstance(spec, spec_type):
                return spec
        assert False, f"no {spec_type} found"

    def test_top_level(self):
        zarr = parse_url(str(self.path))
        assert zarr is not None
        layer = Layer(zarr, list())
        self.matches(layer, {Multiscales, OMERO})
        multiscales = self.get_spec(layer, Multiscales)
        assert multiscales.lookup("multiscales", [])

    def test_labels(self):
        zarr = parse_url(str(self.path + "/labels"))
        assert zarr is not None
        layer = Layer(zarr, list())
        self.matches(layer, set([Labels]))

    def test_label(self):
        zarr = parse_url(str(self.path + "/labels/coins"))
        assert zarr is not None
        layer = Layer(zarr, list())
        self.matches(layer, set([Label, Multiscales]))
        multiscales = self.get_spec(layer, Multiscales)
        assert multiscales.lookup("multiscales", [])
