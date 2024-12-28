import os
from collections import deque
from pathlib import Path

import pytest

from ome_zarr.cli import main
from ome_zarr.utils import strip_common_prefix


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

        out_path = Path(out) / "data-3"
        paths = [p.relative_to(out_path) for p in out_path.glob("*")]
        assert sorted(paths) == [
            Path(".zattrs"),
            Path(".zgroup"),
            Path("0"),
            Path("1"),
            Path("2"),
            Path("3"),
            Path("4"),
            Path("labels"),
        ]

        data_path = Path(out) / "data-3" / "1"
        paths = [p.relative_to(data_path) for p in data_path.glob("*")]
        assert sorted(paths) == [
            Path(".zarray"),
            Path("0.0.0"),
            Path("0.0.1"),
            Path("0.0.2"),
            Path("0.0.3"),
            Path("0.0.4"),
            Path("0.0.5"),
            Path("0.0.6"),
            Path("0.0.7"),
            Path("0.1.0"),
            Path("0.1.1"),
            Path("0.1.2"),
            Path("0.1.3"),
            Path("0.1.4"),
            Path("0.1.5"),
            Path("0.1.6"),
            Path("0.1.7"),
            Path("0.2.0"),
            Path("0.2.1"),
            Path("0.2.2"),
            Path("0.2.3"),
            Path("0.2.4"),
            Path("0.2.5"),
            Path("0.2.6"),
            Path("0.2.7"),
            Path("0.3.0"),
            Path("0.3.1"),
            Path("0.3.2"),
            Path("0.3.3"),
            Path("0.3.4"),
            Path("0.3.5"),
            Path("0.3.6"),
            Path("0.3.7"),
            Path("0.4.0"),
            Path("0.4.1"),
            Path("0.4.2"),
            Path("0.4.3"),
            Path("0.4.4"),
            Path("0.4.5"),
            Path("0.4.6"),
            Path("0.4.7"),
            Path("0.5.0"),
            Path("0.5.1"),
            Path("0.5.2"),
            Path("0.5.3"),
            Path("0.5.4"),
            Path("0.5.5"),
            Path("0.5.6"),
            Path("0.5.7"),
            Path("0.6.0"),
            Path("0.6.1"),
            Path("0.6.2"),
            Path("0.6.3"),
            Path("0.6.4"),
            Path("0.6.5"),
            Path("0.6.6"),
            Path("0.6.7"),
            Path("0.7.0"),
            Path("0.7.1"),
            Path("0.7.2"),
            Path("0.7.3"),
            Path("0.7.4"),
            Path("0.7.5"),
            Path("0.7.6"),
            Path("0.7.7"),
            Path("1.0.0"),
            Path("1.0.1"),
            Path("1.0.2"),
            Path("1.0.3"),
            Path("1.0.4"),
            Path("1.0.5"),
            Path("1.0.6"),
            Path("1.0.7"),
            Path("1.1.0"),
            Path("1.1.1"),
            Path("1.1.2"),
            Path("1.1.3"),
            Path("1.1.4"),
            Path("1.1.5"),
            Path("1.1.6"),
            Path("1.1.7"),
            Path("1.2.0"),
            Path("1.2.1"),
            Path("1.2.2"),
            Path("1.2.3"),
            Path("1.2.4"),
            Path("1.2.5"),
            Path("1.2.6"),
            Path("1.2.7"),
            Path("1.3.0"),
            Path("1.3.1"),
            Path("1.3.2"),
            Path("1.3.3"),
            Path("1.3.4"),
            Path("1.3.5"),
            Path("1.3.6"),
            Path("1.3.7"),
            Path("1.4.0"),
            Path("1.4.1"),
            Path("1.4.2"),
            Path("1.4.3"),
            Path("1.4.4"),
            Path("1.4.5"),
            Path("1.4.6"),
            Path("1.4.7"),
            Path("1.5.0"),
            Path("1.5.1"),
            Path("1.5.2"),
            Path("1.5.3"),
            Path("1.5.4"),
            Path("1.5.5"),
            Path("1.5.6"),
            Path("1.5.7"),
            Path("1.6.0"),
            Path("1.6.1"),
            Path("1.6.2"),
            Path("1.6.3"),
            Path("1.6.4"),
            Path("1.6.5"),
            Path("1.6.6"),
            Path("1.6.7"),
            Path("1.7.0"),
            Path("1.7.1"),
            Path("1.7.2"),
            Path("1.7.3"),
            Path("1.7.4"),
            Path("1.7.5"),
            Path("1.7.6"),
            Path("1.7.7"),
            Path("2.0.0"),
            Path("2.0.1"),
            Path("2.0.2"),
            Path("2.0.3"),
            Path("2.0.4"),
            Path("2.0.5"),
            Path("2.0.6"),
            Path("2.0.7"),
            Path("2.1.0"),
            Path("2.1.1"),
            Path("2.1.2"),
            Path("2.1.3"),
            Path("2.1.4"),
            Path("2.1.5"),
            Path("2.1.6"),
            Path("2.1.7"),
            Path("2.2.0"),
            Path("2.2.1"),
            Path("2.2.2"),
            Path("2.2.3"),
            Path("2.2.4"),
            Path("2.2.5"),
            Path("2.2.6"),
            Path("2.2.7"),
            Path("2.3.0"),
            Path("2.3.1"),
            Path("2.3.2"),
            Path("2.3.3"),
            Path("2.3.4"),
            Path("2.3.5"),
            Path("2.3.6"),
            Path("2.3.7"),
            Path("2.4.0"),
            Path("2.4.1"),
            Path("2.4.2"),
            Path("2.4.3"),
            Path("2.4.4"),
            Path("2.4.5"),
            Path("2.4.6"),
            Path("2.4.7"),
            Path("2.5.0"),
            Path("2.5.1"),
            Path("2.5.2"),
            Path("2.5.3"),
            Path("2.5.4"),
            Path("2.5.5"),
            Path("2.5.6"),
            Path("2.5.7"),
            Path("2.6.0"),
            Path("2.6.1"),
            Path("2.6.2"),
            Path("2.6.3"),
            Path("2.6.4"),
            Path("2.6.5"),
            Path("2.6.6"),
            Path("2.6.7"),
            Path("2.7.0"),
            Path("2.7.1"),
            Path("2.7.2"),
            Path("2.7.3"),
            Path("2.7.4"),
            Path("2.7.5"),
            Path("2.7.6"),
            Path("2.7.7"),
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
