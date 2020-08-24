# -*- coding: utf-8 -*-

import os
import tempfile

from ome_zarr.cli import main


class TestCli:
    @classmethod
    def setup_class(cls):
        cls.path = tempfile.TemporaryDirectory().name

    def test_coins(self):
        filename = os.path.join(self.path, "coins")
        main(["coins", filename])
        main(["info", filename])
