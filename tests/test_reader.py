# -*- coding: utf-8 -*-

import pytest

from ome_zarr.data import create_zarr
from ome_zarr.io import parse_url
from ome_zarr.reader import Layer, Reader


class TestReader:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = tmpdir.mkdir("data")
        create_zarr(str(self.path))

    def assert_layer(self, layer: Layer):
        if not layer.data or not layer.metadata:
            assert False, f"Empty layer received: {layer}"

    def test_image(self):
        reader = Reader(parse_url(str(self.path)))
        for layer in reader():
            self.assert_layer(layer)

    def test_labels(self):
        filename = str(self.path.join("labels"))
        reader = Reader(parse_url(filename))
        for layer in reader():
            self.assert_layer(layer)

    def test_label(self):
        filename = str(self.path.join("labels", "coins"))
        reader = Reader(parse_url(filename))
        for layer in reader():
            self.assert_layer(layer)
