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

    @pytest.mark.parametrize("downsample_z", [True, False])
    @pytest.mark.parametrize("input_group", [True, False])
    def test_writer(self, downsample_z, input_group):

        shape = (1, 2, 16, 256, 256)
        data = self.create_data(shape)
        write_image(image=data, group=self.group, chunks=(128, 128))

        in_path = f"{self.path}/test"
        out_path = f"{self.path}/test_scaled"

        if not input_group:
            in_path += "/0"
            out_path += "_fromArray"
        if downsample_z:
            out_path += "_downZ"
        args = ["scale", in_path, out_path]

        if downsample_z:
            args.append("--downsample_z")

        # If starting with a group (pyramid) and we're not downsampling
        # in Z then there's nothing to do.
        if input_group and downsample_z is False:
            # TODO - check that running main(args) raises error
            return

        main(args)

        size_z = 16
        if downsample_z:
            size_z = 8

        # Verify
        reader = Reader(parse_url(out_path))
        node = list(reader())[0]
        assert Multiscales.matches(node.zarr)
        assert node.data[0].shape == (1, 2, 16, 256, 256)
        assert node.data[1].shape == (1, 2, size_z, 128, 128)
        # assert node.data[0].chunks == ((1,), (2,), (1,), (128, 128), (128, 128))
