from pathlib import Path

import fsspec
import pytest
import zarr

from ome_zarr.data import create_zarr
from ome_zarr.io import ZarrLocation, parse_url


class TestIO:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = tmpdir.mkdir("data")
        create_zarr(str(self.path))
        self.store = parse_url(str(self.path), mode="w").store
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
        fs = fsspec.filesystem("memory")
        fsstore = zarr.storage.FSStore(url="/", fs=fs)
        loc = ZarrLocation(fsstore)
        assert loc
