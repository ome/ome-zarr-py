import sys

import numpy as np
import pytest

from ome_zarr.data import astronaut, create_zarr
from ome_zarr.napari import napari_get_reader


class TestNapari:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = tmpdir.mkdir("data")
        create_zarr(str(self.path), astronaut, "astronaut")

    def assert_layers(self, layers, visible_1, visible_2, label_props=None):
        # TODO: check name

        assert len(layers) == 2
        image, label = layers

        data, metadata, layer_type = self.assert_layer(image)
        assert 1 == metadata["channel_axis"]
        assert ["Red", "Green", "Blue"] == metadata["name"]
        assert [[0, 1]] * 3 == metadata["contrast_limits"]
        assert [visible_1] * 3 == metadata["visible"]

        data, metadata, layer_type = self.assert_layer(label)
        assert visible_2 == metadata["visible"]
        if label_props:
            assert label_props == metadata["properties"]

    def assert_layer(self, layer_data):
        data, metadata, layer_type = layer_data
        if not data or not metadata:
            assert False, f"unknown layer: {layer_data}"
        assert layer_type in ("image", "labels")
        return data, metadata, layer_type

    def test_image(self):
        layers = napari_get_reader(str(self.path))()
        self.assert_layers(layers, True, False)

    def test_labels(self):
        filename = str(self.path.join("labels"))
        layers = napari_get_reader(filename)()
        self.assert_layers(layers, False, True)

    def test_label(self):
        filename = str(self.path.join("labels", "astronaut"))
        layers = napari_get_reader(filename)()
        properties = {
            "index": [i for i in range(1, 9)],
            "class": [f"class {i}" for i in range(1, 9)],
        }
        self.assert_layers(layers, False, True, properties)

    @pytest.mark.qt
    @pytest.mark.skipif(
        sys.version_info < (3, 7),
        reason="on_draw is missing in napari < 0.4.0",
    )
    def test_viewer(self, make_napari_viewer):  # noqa
        """example of testing the viewer."""
        viewer = make_napari_viewer()

        shapes = [(4000, 3000), (2000, 1500), (1000, 750), (500, 375)]
        np.random.seed(0)
        data = [np.random.random(s) for s in shapes]
        _ = viewer.add_image(data, multiscale=True, contrast_limits=[0, 1])
        layer = viewer.layers[0]

        # Set canvas size to target amount
        viewer.window.qt_viewer.view.canvas.size = (800, 600)
        viewer.window.qt_viewer.on_draw(None)

        # Check that current level is first large enough to fill the canvas with
        # a greater than one pixel depth
        assert layer.data_level == 2
