from pathlib import Path

import fsspec
import pytest
import zarr
from zarr.storage import LocalStore

from ome_zarr.data import create_zarr
from ome_zarr.io import ZarrLocation, parse_url


class TestIO:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = tmpdir.mkdir("data")
        create_zarr(str(self.path))
        # this overwrites the data if mode="w"
        self.store = parse_url(str(self.path), mode="r").store
        self.root = zarr.group(store=self.store)

    def test_parse_url(self):
        assert parse_url(str(self.path))

    def test_parse_nonexistent_url(self):
        assert parse_url(self.path + "/does-not-exist") is None

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
