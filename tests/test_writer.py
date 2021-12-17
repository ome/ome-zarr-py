import pathlib

import numpy as np
import pytest
import zarr

from ome_zarr.format import FormatV01, FormatV02, FormatV03, FormatV04
from ome_zarr.io import parse_url
from ome_zarr.reader import Multiscales, Reader
from ome_zarr.scale import Scaler
from ome_zarr.writer import (
    KNOWN_AXES,
    _validate_axes,
    write_image,
    write_multiscales_metadata,
)


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
            pytest.param(FormatV04, id="V04"),
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
        if version.version in ("0.1", "0.2"):
            # v0.1 and v0.2 MUST be 5D
            assert node.data[0].ndim == 5
        else:
            assert node.data[0].shape == shape
        assert np.allclose(data, node.data[0][...].compute())

    def test_dim_names(self):

        v03 = FormatV03()

        # v0.3 MUST specify axes for 3D or 4D data
        with pytest.raises(ValueError):
            _validate_axes(3, axes=None, fmt=v03)

        # ndims must match axes length
        with pytest.raises(ValueError):
            _validate_axes(3, axes="yx", fmt=v03)

        # axes must be ordered tczyx
        with pytest.raises(ValueError):
            _validate_axes(3, axes="yxt", fmt=v03)
        with pytest.raises(ValueError):
            _validate_axes(2, axes=["x", "y"], fmt=v03)
        with pytest.raises(ValueError):
            _validate_axes(5, axes="xyzct", fmt=v03)

        # valid axes - no change, converted to list
        assert _validate_axes(2, axes=["y", "x"], fmt=v03) == ["y", "x"]
        assert _validate_axes(5, axes="tczyx", fmt=v03) == [
            "t",
            "c",
            "z",
            "y",
            "x",
        ]

        # if 2D or 5D, axes can be assigned automatically
        assert _validate_axes(2, axes=None, fmt=v03) == ["y", "x"]
        assert _validate_axes(5, axes=None, fmt=v03) == ["t", "c", "z", "y", "x"]

        # for v0.1 or v0.2, axes should be None
        assert _validate_axes(2, axes=["y", "x"], fmt=FormatV01()) is None
        assert _validate_axes(2, axes=["y", "x"], fmt=FormatV02()) is None

        # check that write_image is checking axes
        data = self.create_data((125, 125))
        with pytest.raises(ValueError):
            write_image(
                image=data,
                group=self.group,
                fmt=v03,
                axes="xyz",
            )

    def test_axes_dicts(self):

        v04 = FormatV04()

        # ALL axes must specify 'name'
        with pytest.raises(ValueError):
            _validate_axes(2, axes=[{"name": "y"}, {}], fmt=v04)

        all_dims = [
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ]

        # auto axes for 2D, 5D, converted to dict for v0.4
        assert _validate_axes(2, axes=None, fmt=v04) == all_dims[-2:]
        assert _validate_axes(5, axes=None, fmt=v04) == all_dims

        # convert from list or string
        assert _validate_axes(3, axes=["z", "y", "x"], fmt=v04) == all_dims[-3:]
        assert _validate_axes(4, axes="czyx", fmt=v04) == all_dims[-4:]

        # invalid based on ordering of types
        with pytest.raises(ValueError):
            assert _validate_axes(3, axes=["y", "c", "x"], fmt=v04)
        with pytest.raises(ValueError):
            assert _validate_axes(4, axes="ctyx", fmt=v04)

        # custom types
        assert _validate_axes(3, axes=["foo", "y", "x"], fmt=v04) == [
            {"name": "foo"},
            all_dims[-2],
            all_dims[-1],
        ]

        # space types can be in ANY order
        assert _validate_axes(3, axes=["x", "z", "y"], fmt=v04) == [
            all_dims[-1],
            all_dims[-3],
            all_dims[-2],
        ]

        # Not allowed multiple custom types
        with pytest.raises(ValueError):
            _validate_axes(4, axes=["foo", "bar", "y", "x"], fmt=v04)

        # unconventional naming is allowed
        strange_axes = [
            {"name": "duration", "type": "time"},
            {"name": "rotation", "type": "angle"},
            {"name": "dz", "type": "space"},
            {"name": "WIDTH", "type": "space"},
        ]
        assert _validate_axes(4, axes=strange_axes, fmt=v04) == strange_axes

        # check that write_image is checking axes
        data = self.create_data((125, 125))
        with pytest.raises(ValueError):
            write_image(
                image=data,
                group=self.group,
                fmt=v04,
                axes="xt",
            )


class TestMultiscalesMetadata:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = pathlib.Path(tmpdir.mkdir("data"))
        self.store = parse_url(self.path, mode="w").store
        self.root = zarr.group(store=self.store)

    def test_single_level(self):
        write_multiscales_metadata(self.root, ["0"])
        assert "multiscales" in self.root.attrs
        assert "version" in self.root.attrs["multiscales"][0]
        assert self.root.attrs["multiscales"][0]["datasets"] == [{"path": "0"}]

    def test_multi_levels(self):
        write_multiscales_metadata(self.root, ["0", "1", "2"])
        assert "multiscales" in self.root.attrs
        assert "version" in self.root.attrs["multiscales"][0]
        assert self.root.attrs["multiscales"][0]["datasets"] == [
            {"path": "0"},
            {"path": "1"},
            {"path": "2"},
        ]

    @pytest.mark.parametrize("fmt", (FormatV01(), FormatV02(), FormatV03()))
    def test_version(self, fmt):
        write_multiscales_metadata(self.root, ["0"], fmt=fmt)
        assert "multiscales" in self.root.attrs
        assert self.root.attrs["multiscales"][0]["version"] == fmt.version
        assert self.root.attrs["multiscales"][0]["datasets"] == [{"path": "0"}]

    @pytest.mark.parametrize(
        "axes",
        (
            ["y", "x"],
            ["c", "y", "x"],
            ["z", "y", "x"],
            ["t", "y", "x"],
            ["t", "c", "y", "x"],
            ["t", "z", "y", "x"],
            ["c", "z", "y", "x"],
            ["t", "c", "z", "y", "x"],
        ),
    )
    def test_axes(self, axes):
        write_multiscales_metadata(self.root, ["0"], axes=axes)
        assert "multiscales" in self.root.attrs
        axes = [{"name": name, "type": KNOWN_AXES[name]} for name in axes]
        assert self.root.attrs["multiscales"][0]["axes"] == axes

    @pytest.mark.parametrize("fmt", (FormatV01(), FormatV02()))
    def test_axes_ignored(self, fmt):
        write_multiscales_metadata(
            self.root, ["0"], fmt=fmt, axes=["t", "c", "z", "y", "x"]
        )
        assert "multiscales" in self.root.attrs
        assert "axes" not in self.root.attrs["multiscales"][0]

    @pytest.mark.parametrize(
        "axes",
        (
            [],
            ["i", "j"],
            ["x", "y"],
            ["y", "x", "c"],
            ["x", "y", "z", "c", "t"],
        ),
    )
    def test_invalid_0_3_axes(self, axes):
        with pytest.raises(ValueError):
            write_multiscales_metadata(self.root, ["0"], fmt=FormatV03(), axes=axes)
