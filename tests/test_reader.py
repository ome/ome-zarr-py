# -*- coding: utf-8 -*-

import os
import tempfile

from ome_zarr.data import create_zarr
from ome_zarr.io import parse_url
from ome_zarr.reader import Layer, Reader


class TestReader:
    @classmethod
    def setup_class(cls):
        cls.path = tempfile.TemporaryDirectory().name
        create_zarr(cls.path)

    def assert_layer(self, layer: Layer):
        if not layer.data or not layer.metadata:
            assert False, f"Empty layer received: {layer}"

    def test_image(self):
        reader = Reader(parse_url(self.path))
        for layer in reader():
            self.assert_layer(layer)

    def test_labels(self):
        filename = os.path.join(self.path + "/labels")
        reader = Reader(parse_url(filename))
        for layer in reader():
            self.assert_layer(layer)

    def test_label(self):
        filename = os.path.join(self.path + "/labels/coins")
        reader = Reader(parse_url(filename))
        for layer in reader():
            self.assert_layer(layer)
