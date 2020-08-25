# -*- coding: utf-8 -*-

import pytest

from ome_zarr.data import astronaut, create_zarr
from ome_zarr.napari import napari_get_reader


class TestNapari:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = tmpdir.mkdir("data")
        create_zarr(str(self.path), astronaut)

    def assert_layer(self, layer_data):
        data, metadata = layer_data
        if not data or not metadata:
            assert False, f"unknown layer: {layer_data}"
        return data, metadata

    def test_image(self):
        layers = napari_get_reader(str(self.path))()
        assert layers
        for layer_data in layers:
            data, metadata = self.assert_layer(layer_data)
            assert 1 == metadata["channel_axis"]
            assert ["Red", "Green", "Blue"] == metadata["name"]
            assert [[0, 1], [0, 1], [0, 1]] == metadata["contrast_limits"]
            assert [True, True, True] == metadata["visible"]

    def test_labels(self):
        filename = str(self.path.join("labels"))
        layers = napari_get_reader(filename)()
        assert layers
        for layer_data in layers:
            data, metadata = self.assert_layer(layer_data)

    def test_label(self):
        filename = str(self.path.join("labels", "coins"))
        layers = napari_get_reader(filename)()
        assert layers
        for layer_data in layers:
            data, metadata = self.assert_layer(layer_data)

    def test_layers(self):
        filename = str(self.path.join("labels", "coins"))
        layers = napari_get_reader(filename)()
        assert layers
        # check order
        # check name
        # check visibility
