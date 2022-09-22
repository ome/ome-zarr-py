import pathlib

import numpy as np
import pytest
import zarr

from ome_zarr.io import parse_url
from ome_zarr.bioformats2raw import bioformats2raw
from ome_zarr.reader import Multiscales, Reader
from ome_zarr.writer import write_image


class TestBf2raw:

    def assert_data(self, path, shape, plate, mode="r"):
        loc = parse_url(path, mode=mode)
        assert loc, f"no zarr found at {path}"
        reader = Reader(loc)
        nodes = list(reader())
        for node in nodes:
            if bioformats2raw.matches(node.zarr):
                pass
            elif Multiscales.matches(node.zarr):
                assert node.data[0].shape == shape
                assert np.max(node.data[0]) > 0
            elif "OME" in str(node.zarr):
                assert "series" in node.metadata
            else:
                raise Exception(node)

    @pytest.mark.parametrize(
        "path,plate",
        (
            ("fake-series-2.zarr", False),
            ("plate-rows-2.zarr", True),
         )
    )
    def test_static_data(self, request, path, plate):
        shape = (1, 1, 1, 512, 512)
        self.assert_data(
            f"{request.fspath.dirname}/data/bf2raw/{path}",
            shape,
            plate
        )
