import os
from collections import deque
from pathlib import Path

import pytest
import zarr

from ome_zarr.cli import main
from ome_zarr.utils import finder, strip_common_prefix, view
from ome_zarr.writer import write_plate_metadata


def directory_items(directory: Path):
    """
    Get all items (files and folders) in a directory, relative to that directory.
    """
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a directory")

    return sorted([p.relative_to(directory) for p in directory.glob("*")])


class TestCli:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = (tmpdir / "data").mkdir()

    @pytest.fixture(params=["0.1", "0.2", "0.3"], ids=["v0.1", "v0.2", "v0.3"])
    def s3_address(self, request):
        urls = {
            "0.1": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001240.zarr",
            "0.2": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.2/6001240.zarr",
            "0.3": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.3/9836842.zarr",
        }
        return urls[request.param]

    def test_coins_info(self):
        filename = str(self.path) + "-1"
        main(["create", "--method=coins", filename])
        main(["info", filename])

    def test_astronaut_info(self):
        filename = str(self.path) + "-2"
        main(["create", "--method=astronaut", filename])
        main(["info", filename])

    def test_astronaut_download(self, tmpdir):
        out = str(tmpdir / "out")
        filename = str(self.path) + "-3"
        basename = os.path.split(filename)[-1]
        main(["create", "--method=astronaut", filename])
        main(["download", filename, f"--output={out}"])
        main(["info", f"{out}/{basename}"])

        assert directory_items(Path(out) / "data-3") == [
            Path(".zattrs"),
            Path(".zgroup"),
            Path("0"),
            Path("1"),
            Path("2"),
            Path("3"),
            Path("4"),
            Path("labels"),
        ]

        assert directory_items(Path(out) / "data-3" / "1") == [
            Path(".zarray"),
            Path("0"),
            Path("1"),
            Path("2"),
        ]

    def test_s3_info(self, s3_address):
        main(["info", s3_address])

    def test_strip_prefix_relative(self):
        top = Path(".") / "d"
        mid = Path(".") / "d" / "e"
        bot = Path(".") / "d" / "e" / "f"
        self._rotate_and_test(top, mid, bot)

    def test_strip_prefix_absolute(self):
        top = Path("/") / "a" / "b" / "c" / "d"
        mid = Path("/") / "a" / "b" / "c" / "d" / "e"
        bot = Path("/") / "a" / "b" / "c" / "d" / "e" / "f"
        self._rotate_and_test(top, mid, bot)

    def _rotate_and_test(self, *hierarchy: Path, reverse: bool = True):
        results: list[list[str]] = [
            list((Path("d")).parts),
            list((Path("d") / "e").parts),
            list((Path("d") / "e" / "f").parts),
        ]
        for x in range(3):
            firstpass = deque(hierarchy)
            firstpass.rotate(1)

            copy = [list(x.parts) for x in firstpass]
            common = strip_common_prefix(copy)
            assert "d" == common
            assert {tuple(x) for x in copy} == {tuple(x) for x in results}

        if reverse:
            secondpass: deque = deque(hierarchy)
            secondpass.reverse()
            self._rotate_and_test(*list(secondpass), reverse=False)

    def test_view(self):
        filename = str(self.path) + "-4"
        main(["create", "--method=astronaut", filename])
        # CLI doesn't support the dry_run option yet
        # main(["view", filename, "8000"])
        # we need dry_run to be True to avoid blocking the test with server
        view(filename, 8000, True)

    def test_finder(self):
        img_dir = (self.path / "images").mkdir()
        img_dir2 = (img_dir / "dir2").mkdir()
        bf2raw_dir = (img_dir / "bf2raw.zarr").mkdir()
        main(["create", "--method=astronaut", (str(img_dir / "astronaut"))])
        main(["create", "--method=coins", (str(img_dir2 / "coins"))])
        (bf2raw_dir / "OME").mkdir()

        # write minimal bioformats2raw and xml metadata
        with open(bf2raw_dir / ".zattrs", "w") as f:
            f.write("""{"bioformats2raw.layout" : 3}""")
        with open(bf2raw_dir / "OME" / "METADATA.ome.xml", "w") as f:
            f.write(
                """<?xml version="1.0" encoding="UTF-8"?>
                <OME><Image ID="Image:1" Name="test.fake"></Image></OME>
                """
            )

        # create a plate
        plate_dir = (img_dir2 / "plate").mkdir()
        store = zarr.DirectoryStore(str(plate_dir))
        root = zarr.group(store=store)
        write_plate_metadata(root, ["A"], ["1"], ["A/1"])

        finder(img_dir, 8000, True)

        assert (img_dir / "biofile_finder.csv").exists()
        csv_text = (img_dir / "biofile_finder.csv").read_text(encoding="utf-8")
        print(csv_text)
        assert "File Path,File Name,Folders,Uploaded" in csv_text
        assert "dir2/plate/A/1/0,plate,dir2" in csv_text
        assert "coins,dir2" in csv_text
        assert "test.fake" in csv_text
