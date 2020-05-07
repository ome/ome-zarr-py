# -*- coding: utf-8 -*-

from ome_zarr import napari_get_reader, info, download
from .create_test_data import create_zarr
import tempfile
import os

class TestOmeZarr:

    @classmethod
    def setup_class(cls):
        """ setup any state specific to the execution of the given class (which
        usually contains tests).
        """
        cls.path = tempfile.TemporaryDirectory(suffix=".zarr").name
        create_zarr(cls.path)

    # @classmethod
    # def teardown_class(cls):
    #     """ teardown any state that was previously setup with a call to
    #     setup_class.
    #     """

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
        assert result[1]['channel_axis'] == 1
        assert result[1]['name'] == ['Red', 'Green', 'Blue']

    def test_get_reader_with_list(self):
        # a better test here would use real data
        reader = napari_get_reader([self.path])
        assert reader is not None
        assert callable(reader)

    def test_get_reader_pass(self):
        reader = napari_get_reader('fake.file')
        assert reader is None

    def check_info_stdout(self, out, check_metadata=True):
        # from print statements in reader
        assert ("resolution 0 shape (t, c, z, y, x) (1, 3, 1, 1024, 1024)"
                " chunks ['1', '1', '1', '256', '256'] dtype float64") in out
        assert ("resolution 1 shape (t, c, z, y, x) (1, 3, 1, 512, 512)"
                " chunks ['1', '1', '1', '256', '256'] dtype float64") in out
        # from info's print of dask array
        assert ("[dask.array<from-zarr, shape=(1, 3, 1, 1024, 1024), dtype=float64,"
                " chunksize=(1, 1, 1, 256, 256), chunktype=numpy.ndarray>,"
                " dask.array<from-zarr, shape=(1, 3, 1, 512, 512), dtype=float64,"
                " chunksize=(1, 1, 1, 256, 256), chunktype=numpy.ndarray>,"
                " dask.array<from-zarr, shape=(1, 3, 1, 256, 256), dtype=float64,"
                " chunksize=(1, 2, 1, 128, 128), chunktype=numpy.ndarray>") in out
        # from info's print of omero json
        if check_metadata:
            assert "'channel_axis': 1" in out
            assert "'name': ['Red', 'Green', 'Blue']" in out
            assert "'contrast_limits': [[0, 1], [0, 1], [0, 1]]" in out
            assert "'visible': [True, True, True]" in out

    def test_info(self, capsys):
        info(self.path)
        out = capsys.readouterr().out
        self.check_info_stdout(out)

    def test_download(self, capsys):
        target = tempfile.TemporaryDirectory().name
        name = 'test.zarr'
        download(self.path, output_dir=target, zarr_name=name)
        download_zarr = os.path.join(target, name)
        assert os.path.exists(download_zarr)
        info(download_zarr)
        out = capsys.readouterr().out
        # omero.json metadata not downloaded
        self.check_info_stdout(out, check_metadata=False)
        # check download progress in stdout
        assert "100% Completed" in out
