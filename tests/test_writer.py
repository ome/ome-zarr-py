import pathlib

import numpy as np
import pytest
import zarr

from ome_zarr.format import FormatV01, FormatV02, FormatV03
from ome_zarr.io import parse_url
from ome_zarr.reader import Multiscales, Reader
from ome_zarr.scale import Scaler
from ome_zarr.writer import validate_axes_names, write_image


class TestWriter:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = pathlib.Path(tmpdir.mkdir("data"))
        self.store = parse_url(self.path, mode="w").store
        self.root = zarr.group(store=self.store)
        self.group = self.root.create_group("test")

    def create_data(self, shape, dtype=np.uint8, mean_val=10):
        rng = np.random.default_rng(0)
        return rng.poisson(mean_val, size=shape).astype(dtype)

    @pytest.fixture(
        params=(
            (1, 2, 1, 256, 256),
            (3, 512, 512),
            (256, 256),
        ),
        ids=["5D", "3D", "2D"],
    )
    def shape(self, request):
        return request.param

    @pytest.fixture(params=[True, False], ids=["scale", "noop"])
    def scaler(self, request):
        if request.param:
            return Scaler()
        else:
            return None

    @pytest.mark.parametrize(
        "format_version",
        (
            pytest.param(
                FormatV01,
                id="V01",
                marks=pytest.mark.xfail(reason="issues with dimension_separator"),
            ),
            pytest.param(FormatV02, id="V02"),
            pytest.param(FormatV03, id="V03"),
        ),
    )
    def test_writer(self, shape, scaler, format_version):

        data = self.create_data(shape)
        version = format_version()
        axes = "tczyx"[-len(shape) :]
        write_image(
            image=data,
            group=self.group,
            chunks=(128, 128),
            scaler=scaler,
            fmt=version,
            axes=axes,
        )

        # Verify
        reader = Reader(parse_url(f"{self.path}/test"))
        node = list(reader())[0]
        assert Multiscales.matches(node.zarr)
        if version.version not in ("0.1", "0.2"):
            # v0.1 and v0.2 MUST be 5D
            assert node.data[0].shape == shape
        else:
            assert node.data[0].ndim == 5
        assert np.allclose(data, node.data[0][...].compute())

    def test_dim_names(self):

        v03 = FormatV03()

        # v0.3 MUST specify axes for 3D or 4D data
        with pytest.raises(ValueError):
            validate_axes_names(3, axes=None, fmt=v03)

        # ndims must match axes length
        with pytest.raises(ValueError):
            validate_axes_names(3, axes="yx", fmt=v03)

        # axes must be ordered tczyx
        with pytest.raises(ValueError):
            validate_axes_names(3, axes="yxt", fmt=v03)
        with pytest.raises(ValueError):
            validate_axes_names(2, axes=["x", "y"], fmt=v03)

        # valid axes
        validate_axes_names(2, axes=["y", "x"], fmt=v03)
        validate_axes_names(5, axes="tczyx", fmt=v03)

        # if 2D or 5D, axes can be assigned automatically
        assert validate_axes_names(2, axes=None, fmt=v03) == ["y", "x"]
        assert validate_axes_names(5, axes=None, fmt=v03) == ["t", "c", "z", "y", "x"]

        # for v0.1 or v0.2, axes should be None
        assert validate_axes_names(2, axes=["y", "x"], fmt=FormatV02()) is None

        # check that write_image is checking axes
        data = self.create_data((125, 125))
        with pytest.raises(ValueError):
            write_image(
                image=data,
                group=self.group,
                fmt=v03,
                axes="xyz",
            )
