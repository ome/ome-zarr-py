import os
from collections import deque
from pathlib import Path

import pytest

from ome_zarr.cli import main
from ome_zarr.utils import strip_common_prefix


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
