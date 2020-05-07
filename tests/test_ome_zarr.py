# -*- coding: utf-8 -*-

from ome_zarr import napari_get_reader, info, download

path = 'https://s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001240.zarr/'


def test_get_reader_hit():
    reader = napari_get_reader(path)
    assert reader is not None
    assert callable(reader)


def test_reader():
    reader = napari_get_reader(path)
    results = reader(path)
    assert results is not None and len(results) == 1
    result = results[0]
    assert isinstance(result[0], list)
    assert isinstance(result[1], dict)
    assert result[1]['channel_axis'] == 1
    assert result[1]['name'] == ['LaminB1', 'Dapi']


def test_get_reader_with_list():
    # a better test here would use real data
    reader = napari_get_reader([path])
    assert reader is not None
    assert callable(reader)


def test_get_reader_pass():
    reader = napari_get_reader('fake.file')
    assert reader is None


def test_info(capsys):
    info(path)
    captured = capsys.readouterr()
    out = captured.out
    # from print statements in reader
    assert "resolution 0 shape (t, c, z, y, x) (1, 2, 236, 275, 271)" in out
    assert "resolution 1 shape (t, c, z, y, x) (1, 2, 236, 137, 135)" in out
    # from info's print of dask array
    assert ("dask.array<from-zarr, shape=(1, 2, 236, 275, 271), dtype=>u2,"
            " chunksize=(1, 1, 1, 275, 271), chunktype=numpy.ndarray>,"
            " dask.array<from-zarr, shape=(1, 2, 236, 137, 135), dtype=>u2,"
            " chunksize=(1, 1, 1, 137, 135), chunktype=numpy.ndarray>]") in out
    # from info's print of omero json
    assert "'channel_axis': 1" in out
    assert "'name': ['LaminB1', 'Dapi']" in out
    assert "'contrast_limits': [[0, 1500], [0, 1500]]" in out
    assert "'visible': [True, True]" in out