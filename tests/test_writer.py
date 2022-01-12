import pathlib

import numpy as np
import pytest
import zarr

from ome_zarr.axes import KNOWN_AXES
from ome_zarr.format import CurrentFormat, FormatV01, FormatV02, FormatV03, FormatV04
from ome_zarr.io import parse_url
from ome_zarr.reader import Multiscales, Reader
from ome_zarr.scale import Scaler
from ome_zarr.writer import (
    validate_axes,
    write_image,
    write_multiscales_metadata,
    write_plate_metadata,
    write_well_metadata,
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
            validate_axes(3, axes=None, fmt=v03)

        # ndims must match axes length
        with pytest.raises(ValueError):
            validate_axes(3, axes="yx", fmt=v03)

        # axes must be ordered tczyx
        with pytest.raises(ValueError):
            validate_axes(3, axes="yxt", fmt=v03)
        with pytest.raises(ValueError):
            validate_axes(2, axes=["x", "y"], fmt=v03)
        with pytest.raises(ValueError):
            validate_axes(5, axes="xyzct", fmt=v03)

        # valid axes - no change, converted to list
        assert validate_axes(2, axes=["y", "x"], fmt=v03) == ["y", "x"]
        assert validate_axes(5, axes="tczyx", fmt=v03) == [
            "t",
            "c",
            "z",
            "y",
            "x",
        ]

        # if 2D or 5D, axes can be assigned automatically
        assert validate_axes(2, axes=None, fmt=v03) == ["y", "x"]
        assert validate_axes(5, axes=None, fmt=v03) == ["t", "c", "z", "y", "x"]

        # for v0.1 or v0.2, axes should be None
        assert validate_axes(2, axes=["y", "x"], fmt=FormatV01()) is None
        assert validate_axes(2, axes=["y", "x"], fmt=FormatV02()) is None

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
            validate_axes(2, axes=[{"name": "y"}, {}], fmt=v04)

        all_dims = [
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ]

        # auto axes for 2D, 5D, converted to dict for v0.4
        assert validate_axes(2, axes=None, fmt=v04) == all_dims[-2:]
        assert validate_axes(5, axes=None, fmt=v04) == all_dims

        # convert from list or string
        assert validate_axes(3, axes=["z", "y", "x"], fmt=v04) == all_dims[-3:]
        assert validate_axes(4, axes="czyx", fmt=v04) == all_dims[-4:]

        # invalid based on ordering of types
        with pytest.raises(ValueError):
            assert validate_axes(3, axes=["y", "c", "x"], fmt=v04)
        with pytest.raises(ValueError):
            assert validate_axes(4, axes="ctyx", fmt=v04)

        # custom types
        assert validate_axes(3, axes=["foo", "y", "x"], fmt=v04) == [
            {"name": "foo"},
            all_dims[-2],
            all_dims[-1],
        ]

        # space types can be in ANY order
        assert validate_axes(3, axes=["x", "z", "y"], fmt=v04) == [
            all_dims[-1],
            all_dims[-3],
            all_dims[-2],
        ]

        # Not allowed multiple custom types
        with pytest.raises(ValueError):
            validate_axes(4, axes=["foo", "bar", "y", "x"], fmt=v04)

        # unconventional naming is allowed
        strange_axes = [
            {"name": "duration", "type": "time"},
            {"name": "rotation", "type": "angle"},
            {"name": "dz", "type": "space"},
            {"name": "WIDTH", "type": "space"},
        ]
        assert validate_axes(4, axes=strange_axes, fmt=v04) == strange_axes

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


class TestPlateMetadata:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = pathlib.Path(tmpdir.mkdir("data"))
        self.store = parse_url(self.path, mode="w").store
        self.root = zarr.group(store=self.store)

    def test_minimal_plate(self):
        write_plate_metadata(self.root, ["A"], ["1"], ["A/1"])
        assert "plate" in self.root.attrs
        assert self.root.attrs["plate"]["columns"] == [{"name": "1"}]
        assert self.root.attrs["plate"]["rows"] == [{"name": "A"}]
        assert self.root.attrs["plate"]["version"] == CurrentFormat().version
        assert self.root.attrs["plate"]["wells"] == [{"path": "A/1"}]
        assert "name" not in self.root.attrs["plate"]
        assert "field_count" not in self.root.attrs["plate"]
        assert "acquisitions" not in self.root.attrs["plate"]

    def test_12wells_plate(self):
        rows = ["A", "B", "C", "D"]
        cols = ["1", "2", "3"]
        wells = [
            "A/1",
            "A/2",
            "A/3",
            "B/1",
            "B/2",
            "B/3",
            "C/1",
            "C/2",
            "C/3",
            "D/1",
            "D/2",
            "D/3",
        ]
        write_plate_metadata(self.root, rows, cols, wells)
        assert "plate" in self.root.attrs
        assert self.root.attrs["plate"]["columns"] == [
            {"name": "1"},
            {"name": "2"},
            {"name": "3"},
        ]
        assert self.root.attrs["plate"]["rows"] == [
            {"name": "A"},
            {"name": "B"},
            {"name": "C"},
            {"name": "D"},
        ]
        assert self.root.attrs["plate"]["version"] == CurrentFormat().version
        assert self.root.attrs["plate"]["wells"] == [
            {"path": "A/1"},
            {"path": "A/2"},
            {"path": "A/3"},
            {"path": "B/1"},
            {"path": "B/2"},
            {"path": "B/3"},
            {"path": "C/1"},
            {"path": "C/2"},
            {"path": "C/3"},
            {"path": "D/1"},
            {"path": "D/2"},
            {"path": "D/3"},
        ]
        assert "name" not in self.root.attrs["plate"]
        assert "field_count" not in self.root.attrs["plate"]
        assert "acquisitions" not in self.root.attrs["plate"]

    @pytest.mark.parametrize("fmt", (FormatV01(), FormatV02(), FormatV03()))
    def test_plate_version(self, fmt):
        write_plate_metadata(self.root, ["A"], ["1"], ["A/1"], fmt=fmt)
        assert "plate" in self.root.attrs
        assert self.root.attrs["plate"]["columns"] == [{"name": "1"}]
        assert self.root.attrs["plate"]["rows"] == [{"name": "A"}]
        assert self.root.attrs["plate"]["version"] == fmt.version
        assert self.root.attrs["plate"]["wells"] == [{"path": "A/1"}]
        assert "name" not in self.root.attrs["plate"]
        assert "field_count" not in self.root.attrs["plate"]
        assert "acquisitions" not in self.root.attrs["plate"]

    def test_plate_name(self):
        write_plate_metadata(self.root, ["A"], ["1"], ["A/1"], name="test")
        assert "plate" in self.root.attrs
        assert self.root.attrs["plate"]["columns"] == [{"name": "1"}]
        assert self.root.attrs["plate"]["name"] == "test"
        assert self.root.attrs["plate"]["rows"] == [{"name": "A"}]
        assert self.root.attrs["plate"]["version"] == CurrentFormat().version
        assert self.root.attrs["plate"]["wells"] == [{"path": "A/1"}]
        assert "field_count" not in self.root.attrs["plate"]
        assert "acquisitions" not in self.root.attrs["plate"]

    def test_field_count(self):
        write_plate_metadata(self.root, ["A"], ["1"], ["A/1"], field_count=10)
        assert "plate" in self.root.attrs
        assert self.root.attrs["plate"]["columns"] == [{"name": "1"}]
        assert self.root.attrs["plate"]["field_count"] == 10
        assert self.root.attrs["plate"]["rows"] == [{"name": "A"}]
        assert self.root.attrs["plate"]["version"] == CurrentFormat().version
        assert self.root.attrs["plate"]["wells"] == [{"path": "A/1"}]
        assert "name" not in self.root.attrs["plate"]
        assert "acquisitions" not in self.root.attrs["plate"]

    def test_acquisitions_minimal(self):
        a = [{"id": 1}, {"id": 2}, {"id": 3}]
        write_plate_metadata(self.root, ["A"], ["1"], ["A/1"], acquisitions=a)
        assert "plate" in self.root.attrs
        assert self.root.attrs["plate"]["acquisitions"] == a
        assert self.root.attrs["plate"]["columns"] == [{"name": "1"}]
        assert self.root.attrs["plate"]["rows"] == [{"name": "A"}]
        assert self.root.attrs["plate"]["version"] == CurrentFormat().version
        assert self.root.attrs["plate"]["wells"] == [{"path": "A/1"}]
        assert "name" not in self.root.attrs["plate"]
        assert "field_count" not in self.root.attrs["plate"]

    def test_acquisitions_maximal(self):
        a = [
            {
                "id": 1,
                "name": "acquisition_1",
                "description": " first acquisition",
                "maximumfieldcount": 2,
                "starttime": 1343749391000,
                "endtime": 1343749392000,
            }
        ]
        write_plate_metadata(self.root, ["A"], ["1"], ["A/1"], acquisitions=a)
        assert "plate" in self.root.attrs
        assert self.root.attrs["plate"]["acquisitions"] == a
        assert self.root.attrs["plate"]["columns"] == [{"name": "1"}]
        assert self.root.attrs["plate"]["rows"] == [{"name": "A"}]
        assert self.root.attrs["plate"]["version"] == CurrentFormat().version
        assert self.root.attrs["plate"]["wells"] == [{"path": "A/1"}]
        assert "name" not in self.root.attrs["plate"]
        assert "field_count" not in self.root.attrs["plate"]

    @pytest.mark.parametrize(
        "acquisitions",
        (
            [0, 1],
            [{"name": "0"}, {"name": "1"}],
            [{"id": 0, "invalid_key": "0"}],
            [{"id": "0"}, {"id": "1"}],
        ),
    )
    def test_unspecified_acquisition_keys(self, acquisitions):
        a = [{"id": 0, "invalid_key": "0"}]
        write_plate_metadata(self.root, ["A"], ["1"], ["A/1"], acquisitions=a)
        assert "plate" in self.root.attrs


class TestWellMetadata:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = pathlib.Path(tmpdir.mkdir("data"))
        self.store = parse_url(self.path, mode="w").store
        self.root = zarr.group(store=self.store)

    @pytest.mark.parametrize("images", (["0"], [{"path": "0"}]))
    def test_minimal_well(self, images):
        write_well_metadata(self.root, images)
        assert "well" in self.root.attrs
        assert self.root.attrs["well"]["images"] == [{"path": "0"}]
        assert self.root.attrs["well"]["version"] == CurrentFormat().version

    @pytest.mark.parametrize(
        "images",
        (
            ["0", "1", "2"],
            [
                {"path": "0"},
                {"path": "1"},
                {"path": "2"},
            ],
        ),
    )
    def test_multiple_images(self, images):
        write_well_metadata(self.root, images)
        assert "well" in self.root.attrs
        assert self.root.attrs["well"]["images"] == [
            {"path": "0"},
            {"path": "1"},
            {"path": "2"},
        ]
        assert self.root.attrs["well"]["version"] == CurrentFormat().version

    @pytest.mark.parametrize("fmt", (FormatV01(), FormatV02(), FormatV03()))
    def test_version(self, fmt):
        write_well_metadata(self.root, ["0"], fmt=fmt)
        assert "well" in self.root.attrs
        assert self.root.attrs["well"]["images"] == [{"path": "0"}]
        assert self.root.attrs["well"]["version"] == fmt.version

    def test_multiple_acquisitions(self):
        images = [
            {"path": "0", "acquisition": 1},
            {"path": "1", "acquisition": 2},
            {"path": "2", "acquisition": 3},
        ]
        write_well_metadata(self.root, images)
        assert "well" in self.root.attrs
        assert self.root.attrs["well"]["images"] == images
        assert self.root.attrs["well"]["version"] == CurrentFormat().version

    @pytest.mark.parametrize(
        "images",
        (
            [{"acquisition": 0}, {"acquisition": 1}],
            [{"path": "0", "acquisition": "0"}, {"path": "1", "acquisition": "1"}],
            [{"path": 0}, {"path": 1}],
            [0, 1],
        ),
    )
    def test_invalid_images(self, images):
        with pytest.raises(ValueError):
            write_well_metadata(self.root, images)

    def test_unspecified_images_keys(self):
        images = [
            {"path": "0", "acquisition": 1, "unspecified_key": "alpha"},
            {"path": "1", "acquisition": 2, "unspecified_key": "beta"},
            {"path": "2", "acquisition": 3, "unspecified_key": "gamma"},
        ]
        write_well_metadata(self.root, images)
        assert "well" in self.root.attrs
        assert self.root.attrs["well"]["images"] == images
        assert self.root.attrs["well"]["version"] == CurrentFormat().version
