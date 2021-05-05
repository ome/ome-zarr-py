import pytest

from ome_zarr.cli import main
from ome_zarr.io import parse_url
from ome_zarr.reader import Multiscales, Reader

# from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image

from .test_writer import TestWriter


class TestScaler(TestWriter):

    # @pytest.fixture(autouse=True)
    # def initdir(self, tmpdir):
    #     self.path = pathlib.Path(tmpdir.mkdir("data"))
    #     self.store = parse_url(self.path, mode="w").store
    #     self.root = zarr.group(store=self.store)
    #     self.group = self.root.create_group("test")

    @pytest.mark.parametrize("downsample_z", [True])
    def test_writer(self, downsample_z):

        shape = (1, 2, 16, 256, 256)
        data = self.create_data(shape)
        write_image(image=data, group=self.group, chunks=(128, 128))

        if downsample_z:
            in_path = f"{self.path}/test"
            # Create a new pyramid here, with downsampled Z
            out_path = f"{self.path}/test_scaled"
            args = ["scale", in_path, out_path, "--downsample_z"]
        else:
            # Create a pyramid from a single array
            in_path = f"{self.path}/test/0"
            out_path = f"{self.path}/test_scaled"
            args = ["scale", in_path, out_path]

        main(args)

        # Verify
        reader = Reader(parse_url(out_path))
        node = list(reader())[0]
        assert Multiscales.matches(node.zarr)
        assert node.data[0].shape == (1, 2, 16, 256, 256)
        assert node.data[1].shape == (1, 2, 8, 128, 128)
        # assert node.data[0].chunks == ((1,), (2,), (1,), (128, 128), (128, 128))
