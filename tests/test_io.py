import zipfile
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore, MemoryStore, StorePath, ZipStore

from ome_zarr import OMEZarrImage, OMEZarrMultiscale
from ome_zarr.data import create_zarr
from ome_zarr.io import ZarrLocation, parse_url
from ome_zarr.reader import Reader
from ome_zarr.writer import add_metadata, get_metadata, write_image


class TestIO:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = tmpdir.mkdir("data")
        create_zarr(str(self.path))
        self.store = parse_url(str(self.path), mode="r").store
        self.root = zarr.open_group(store=self.store, mode="r")

    def test_parse_url(self):
        assert parse_url(str(self.path))

    def test_parse_nonexistent_url(self):
        assert parse_url(str(self.path + "/does-not-exist")) is None

    def test_loc_str(self):
        assert ZarrLocation(str(self.path))

    def test_loc_path(self):
        assert ZarrLocation(Path(self.path))

    def test_loc_store(self):
        assert ZarrLocation(self.store)

    def test_loc_fs(self):
        store = LocalStore(str(self.path))
        loc = ZarrLocation(store)
        assert loc

    def test_no_overwrite(self):
        print("self.path:", self.path)

        assert self.root.attrs.get("ome") is not None
        # Test that we can open a store to write, without
        # overwriting existing data
        new_store = parse_url(str(self.path), mode="w").store
        new_root = zarr.open_group(store=new_store)
        add_metadata(new_root, {"extra": "test_no_overwrite"})
        # read...
        read_store = parse_url(str(self.path)).store
        read_root = zarr.open_group(store=read_store, mode="r")
        attrs = get_metadata(read_root)
        assert attrs.get("extra") == "test_no_overwrite"
        assert attrs.get("multiscales") is not None


# An image to write on a store that has no path of its own.
IMAGE = np.arange(64 * 64, dtype="uint16").reshape(64, 64)


def test_store_without_a_path():
    store = MemoryStore()
    write_image(IMAGE, zarr.open_group(store, mode="w"), axes="yx", scaler=None)

    loc = ZarrLocation(store)
    assert loc.exists()
    assert loc.store is store
    nodes = list(Reader(loc)())
    assert nodes, "the reader found no node on the store"
    np.testing.assert_array_equal(np.asarray(nodes[0].data[0]), IMAGE)


def test_prefix_inside_a_store():
    store = MemoryStore()
    write_image(
        IMAGE,
        zarr.open_group(store, path="images/img", mode="w"),
        axes="yx",
        scaler=None,
    )

    loc = ZarrLocation(StorePath(store, "images/img"))
    assert loc.exists()
    assert loc.basename() == "img"
    finest = loc.root_attrs["multiscales"][0]["datasets"][0]["path"]
    np.testing.assert_array_equal(np.asarray(loc.load(finest)), IMAGE)

    # A child location shares the store and extends the prefix.
    assert loc.create(finest) == ZarrLocation(StorePath(store, f"images/img/{finest}"))
    assert not ZarrLocation(StorePath(store, "images/other")).exists()


def test_unrelated_stores_differ():
    assert ZarrLocation(MemoryStore()) != ZarrLocation(MemoryStore())


def test_class_writer_on_a_store():
    """The class-based writer and ZarrLocation share a store without a path."""
    store = MemoryStore()
    multiscale = OMEZarrMultiscale(
        image=OMEZarrImage(data=IMAGE, axes="yx"), scale_factors=None, method=None
    )
    multiscale.to_ome_zarr(
        zarr.open_group(store, path="images/img", mode="w"), overwrite=True
    )

    loc = ZarrLocation(StorePath(store, "images/img"))
    assert loc.exists()
    nodes = list(Reader(loc)())
    assert nodes, "the reader found no node on the store"
    np.testing.assert_array_equal(np.asarray(nodes[0].data[0]), IMAGE)

    read_back = OMEZarrMultiscale.from_ome_zarr(
        zarr.open_group(store, path="images/img", mode="r")
    )
    np.testing.assert_array_equal(np.asarray(read_back.images[0].data), IMAGE)


def test_read_from_a_zip(tmp_path):
    """A zipped OME-Zarr reads through ZarrLocation on a ZipStore."""
    # Dask writes to a ZipStore corrupt the archive (zarr-python#3516),
    # so the test zips a directory store instead of writing to the ZipStore.
    directory = tmp_path / "img.zarr"
    write_image(IMAGE, zarr.open_group(directory, mode="w"), axes="yx", scaler=None)
    archive = tmp_path / "img.zarr.zip"
    with zipfile.ZipFile(archive, mode="w") as zf:
        for file in directory.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(directory).as_posix())

    store = ZipStore(archive, mode="r")
    loc = ZarrLocation(store)
    assert loc.exists()
    nodes = list(Reader(loc)())
    assert nodes, "the reader found no node in the archive"
    np.testing.assert_array_equal(np.asarray(nodes[0].data[0]), IMAGE)
    store.close()
