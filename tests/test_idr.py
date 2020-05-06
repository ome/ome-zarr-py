# -*- coding: utf-8 -*-

from ome_zarr import napari_get_reader

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
