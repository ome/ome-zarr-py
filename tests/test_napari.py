# -*- coding: utf-8 -*-

import os
import tempfile

from ome_zarr.data import astronaut, create_zarr
from ome_zarr.napari import napari_get_reader


class TestNapari:
    @classmethod
    def setup_class(cls):
        cls.path = tempfile.TemporaryDirectory().name
        create_zarr(cls.path, astronaut)

    def test_image(self):
        filename = os.path.join(self.path)
        transform = napari_get_reader(filename)
        for layer_data in transform():
            data, metadata = layer_data
            assert data
            assert metadata
            assert 1 == metadata["channel_axis"]
            assert ["Red", "Green", "Blue"] == metadata["name"]
            assert [[0, 1], [0, 1], [0, 1]] == metadata["contrast_limits"]
            assert [True, True, True] == metadata["visible"]

    def test_labels(self):
        filename = os.path.join(self.path + "/labels")
        transform = napari_get_reader(filename)
        for layer_data in transform():
            data, metadata = layer_data
            assert data
            assert metadata

    def test_label(self):
        filename = os.path.join(self.path + "/labels/coins")
        transform = napari_get_reader(filename)
        for layer_data in transform():
            data, metadata = layer_data
            assert data
            assert metadata
