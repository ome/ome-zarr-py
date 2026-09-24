from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore, MemoryStore, StorePath

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


class TestStoreLocation:
    """A location on a store that has no path of its own."""

    image = np.arange(64 * 64, dtype="uint16").reshape(64, 64)

    def test_store_without_a_path(self):
        store = MemoryStore()
        write_image(
            self.image, zarr.open_group(store, mode="w"), axes="yx", scaler=None
        )

        loc = ZarrLocation(store)
        assert loc.exists()
        assert loc.store is store
        nodes = list(Reader(loc)())
        assert nodes, "the reader found no node on the store"
        np.testing.assert_array_equal(np.asarray(nodes[0].data[0]), self.image)

    def test_prefix_inside_a_store(self):
        store = MemoryStore()
        write_image(
            self.image,
            zarr.open_group(store, path="images/img", mode="w"),
            axes="yx",
            scaler=None,
        )

        loc = ZarrLocation(StorePath(store, "images/img"))
        assert loc.exists()
        assert loc.basename() == "img"
        finest = loc.root_attrs["multiscales"][0]["datasets"][0]["path"]
        np.testing.assert_array_equal(np.asarray(loc.load(finest)), self.image)

        # A child location shares the store and extends the prefix.
        assert loc.create(finest) == ZarrLocation(
            StorePath(store, f"images/img/{finest}")
        )
        assert not ZarrLocation(StorePath(store, "images/other")).exists()

    def test_unrelated_stores_differ(self):
        assert ZarrLocation(MemoryStore()) != ZarrLocation(MemoryStore())
