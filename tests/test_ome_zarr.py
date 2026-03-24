import json
import logging
import os
from pathlib import Path

import pytest

from ome_zarr.data import astronaut, create_zarr
from ome_zarr.format import FormatV05
from ome_zarr.utils import download, info


def log_strings(idx, c, y, x, cc, cy, cx, dtype):
    yield f"resolution: s{idx}"
    yield f" - shape ('c', 'y', 'x') = ({c}, {y}, {x})"
    yield f" - chunks =  ['{cc}', '{cx}', '{cy}']"
    yield f" - dtype = {dtype}"


class TestOmeZarr:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = tmpdir.mkdir("data")
        create_zarr(str(self.path), method=astronaut)

    def check_info_stdout(self, out):
        for log in log_strings(0, 3, 1024, 1024, 1, 64, 64, "uint8"):
            assert log in out
        for log in log_strings(1, 3, 512, 512, 1, 64, 64, "int8"):
            assert log in out

        # from info's print of omero metadata
        # note: some metadata is no longer handled by info but rather
        #       in the ome_zarr.napari.transform method

    def test_info(self, caplog):
        with caplog.at_level(logging.DEBUG):
            list(info(str(self.path)))
        self.check_info_stdout(caplog.text)

    def test_download(self, capsys, caplog, tmpdir):
        target = str(tmpdir / "out")
        with caplog.at_level(logging.DEBUG):
            download(str(self.path), output_dir=target)
            download_zarr = os.path.join(target, "data")
            assert os.path.exists(download_zarr)
            info(download_zarr)
        self.check_info_stdout(caplog.text)
        # check download progress in stdout
        out = capsys.readouterr().out
        assert "100% Completed" in out

    def test_download_v3_preserves_root_zarr_json(self, tmpdir):
        source = tmpdir / "data-v3"
        create_zarr(str(source), method=astronaut, fmt=FormatV05())

        target = tmpdir / "out"
        download(str(source), output_dir=str(target))

        root_json = json.loads(
            (Path(target) / "data-v3" / "zarr.json").read_text(encoding="utf-8")
        )

        assert root_json["node_type"] == "group"
        assert root_json["zarr_format"] == 3
        assert "attributes" not in root_json["attributes"]
        assert "node_type" not in root_json["attributes"]
        assert "zarr_format" not in root_json["attributes"]
        assert "ome" in root_json["attributes"]
        assert root_json["attributes"]["ome"]["version"] == FormatV05().version
