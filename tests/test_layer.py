# -*- coding: utf-8 -*-

import os
import tempfile

from ome_zarr.data import create_zarr
from ome_zarr.io import parse_url
from ome_zarr.reader import Layer


class TestLayer:
    @classmethod
    def setup_class(cls):
        cls.path = tempfile.TemporaryDirectory().name
        create_zarr(cls.path)

    def test_image(self):
        layer = Layer(parse_url(self.path))
        assert layer.data
        assert layer.metadata

    def test_labels(self):
        filename = os.path.join(self.path + "/labels")
        layer = Layer(parse_url(filename))
        assert not layer.data
        assert not layer.metadata

    def test_label(self):
        filename = os.path.join(self.path + "/labels/coins")
        layer = Layer(parse_url(filename))
        assert layer.data
        assert layer.metadata
