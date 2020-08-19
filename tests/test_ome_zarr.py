# -*- coding: utf-8 -*-

from ome_zarr.napari import napari_get_reader
from ome_zarr.utils import info, download
from .create_test_data import create_zarr
import tempfile
import os
import logging


def log_strings(idx, t, c, z, y, x, ct, cc, cz, cy, cx, dtype):
    yield f"resolution: {idx}"
    yield f" - shape (t, c, z, y, x) = ({t}, {c}, {z}, {y}, {x})"
    yield f" - chunks =  ['{ct}', '{cc}', '{cz}', '{cx}', '{cy}']"
    yield f" - dtype = {dtype}"


class TestOmeZarr:
    @classmethod
    def setup_class(cls):
        """ setup any state specific to the execution of the given class (which
        usually contains tests).
        """
        cls.path = tempfile.TemporaryDirectory(suffix=".zarr").name
        create_zarr(cls.path)

    def test_get_reader_hit(self):
        reader = napari_get_reader(self.path)
        assert reader is not None
        assert callable(reader)

    def test_reader(self):
        reader = napari_get_reader(self.path)
        results = reader(self.path)
        assert results is not None and len(results) == 1
        result = results[0]
        assert isinstance(result[0], list)
        assert isinstance(result[1], dict)
        assert result[1]["channel_axis"] == 1
        assert result[1]["name"] == ["Red", "Green", "Blue"]

    def test_get_reader_with_list(self):
        # a better test here would use real data
        reader = napari_get_reader([self.path])
        assert reader is not None
        assert callable(reader)

    def test_get_reader_pass(self):
        reader = napari_get_reader("fake.file")
        assert reader is None

    def check_info_stdout(self, out):
        for log in log_strings(0, 1, 3, 1, 1024, 1024, 1, 1, 1, 256, 256, "float64"):
            assert log in out
        for log in log_strings(1, 1, 3, 1, 512, 512, 1, 1, 1, 256, 256, "float64"):
            assert log in out

        # from info's print of omero metadata
        assert "'channel_axis': 1" in out
        assert "'name': ['Red', 'Green', 'Blue']" in out
        assert "'contrast_limits': [[0, 1], [0, 1], [0, 1]]" in out
        assert "'visible': [True, True, True]" in out

    def test_info(self, capsys, caplog):
        with caplog.at_level(logging.DEBUG):
            info(self.path)
        self.check_info_stdout(caplog.text)

    def test_download(self, capsys, caplog):
        target = tempfile.TemporaryDirectory().name
        name = "test.zarr"
        with caplog.at_level(logging.DEBUG):
            download(self.path, output_dir=target, zarr_name=name)
            download_zarr = os.path.join(target, name)
            assert os.path.exists(download_zarr)
            info(download_zarr)
        self.check_info_stdout(caplog.text)
        # check download progress in stdout
        out = capsys.readouterr().out
        assert "100% Completed" in out
