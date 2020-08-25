# -*- coding: utf-8 -*-

import pytest

from ome_zarr.cli import main


class TestCli:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = tmpdir.mkdir("data")

    def test_coins(self):
        filename = str(self.path)
        main(["create", "--method=coins", filename])
        main(["info", filename])

    def test_astronaut(self):
        filename = str(self.path)
        main(["create", "--method=astronaut", filename])
        main(["info", filename])
