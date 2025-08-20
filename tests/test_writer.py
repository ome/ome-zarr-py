import filecmp
import json
import pathlib
from typing import Any

import dask.array as da
import numpy as np
import pytest
import zarr
from dask import persist
from numcodecs import Blosc
from zarr.abc.codec import BytesBytesCodec
from zarr.codecs import BloscCodec

from ome_zarr.format import (
    CurrentFormat,
    FormatV01,
    FormatV02,
    FormatV03,
    FormatV04,
    FormatV05,
)
from ome_zarr.io import ZarrLocation, parse_url
from ome_zarr.reader import Multiscales, Reader
from ome_zarr.scale import Scaler
from ome_zarr.writer import (
    _get_valid_axes,
    _retuple,
    write_image,
    write_labels,
    write_multiscale,
    write_multiscale_labels,
    write_multiscales_metadata,
    write_plate_metadata,
    write_well_metadata,
)

TRANSFORMATIONS = [
    [{"scale": [1, 1, 0.5, 0.18, 0.18], "type": "scale"}],
    [{"scale": [1, 1, 0.5, 0.36, 0.36], "type": "scale"}],
    [{"scale": [1, 1, 0.5, 0.72, 0.72], "type": "scale"}],
    [{"scale": [1, 1, 0.5, 1.44, 1.44], "type": "scale"}],
    [{"scale": [1, 1, 0.5, 2.88, 2.88], "type": "scale"}],
]


class TestWriter:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = pathlib.Path(tmpdir.mkdir("data"))
        # create zarr v2 group...
        self.store = parse_url(self.path, mode="w", fmt=FormatV04()).store
        self.root = zarr.group(store=self.store)
        self.group = self.root.create_group("test")

        # let's create zarr v3 group too...
        self.path_v3 = self.path / "v3"
        store_v3 = parse_url(self.path_v3, mode="w").store
        root_v3 = zarr.group(store=store_v3)
        self.group_v3 = root_v3.create_group("test")

    def create_data(self, shape, dtype=np.uint8, mean_val=10):
        rng = np.random.default_rng(0)
        return rng.poisson(mean_val, size=shape).astype(dtype)

    @pytest.fixture(
        params=(
            (1, 2, 1, 256, 256),
            (3, 512, 512),
            (300, 500),  # test edge chunks of different shapes
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
            pytest.param(FormatV01, id="V01"),
            pytest.param(FormatV02, id="V02"),
            pytest.param(FormatV03, id="V03"),
            pytest.param(FormatV04, id="V04"),
            pytest.param(FormatV05, id="V05"),
        ),
    )
    @pytest.mark.parametrize("array_constructor", [np.array, da.from_array])
    @pytest.mark.parametrize("storage_options_list", [True, False])
    def test_writer(
        self, shape, scaler, format_version, array_constructor, storage_options_list
    ):
        version = format_version()

        if version.version == "0.5":
            group = self.group_v3
            grp_path = self.path_v3 / "test"
        else:
            group = self.group
            grp_path = self.path / "test"

        data = self.create_data(shape)
        data = array_constructor(data)
        axes = "tczyx"[-len(shape) :]
        transformations = []
        for dataset_transfs in TRANSFORMATIONS:
            transf = dataset_transfs[0]
            # e.g. slice [1, 1, z, x, y] -> [z, x, y] for 3D
            transformations.append(
                [{"type": "scale", "scale": transf["scale"][-len(shape) :]}]
            )
        if scaler is None:
            transformations = [transformations[0]]
        chunks = [(128, 128), (50, 50), (25, 25), (25, 25), (25, 25), (25, 25)]
        storage_options = {"chunks": chunks[0]}
        if storage_options_list:
            storage_options = [{"chunks": chunk} for chunk in chunks]
        write_image(
            image=data,
            group=group,
            scaler=scaler,
            fmt=version,
            axes=axes,
            coordinate_transformations=transformations,
            storage_options=storage_options,
        )

        # Verify
        reader = Reader(parse_url(f"{grp_path}"))
        node = next(iter(reader()))
        assert Multiscales.matches(node.zarr)
        if version.version in ("0.1", "0.2"):
            # v0.1 and v0.2 MUST be 5D
            assert node.data[0].ndim == 5
        else:
            assert node.data[0].shape == shape
        print("node.metadata", node.metadata)
        if version.version not in ("0.1", "0.2", "0.3"):
            for transf, expected in zip(
                node.metadata["coordinateTransformations"], transformations
            ):
                assert transf == expected
            assert len(node.metadata["coordinateTransformations"]) == len(node.data)
        # check chunks for first 2 resolutions (before shape gets smaller than chunk)
        for level, nd_array in enumerate(node.data[:2]):
            expected = chunks[level] if storage_options_list else chunks[0]
            first_chunk = [c[0] for c in nd_array.chunks]
            assert tuple(first_chunk) == _retuple(expected, nd_array.shape)
        assert np.allclose(data, node.data[0][...].compute())

    def test_mix_zarr_formats(self):
        # check group zarr v2 and v3 matches fmt
        data = self.create_data((64, 64, 64))
        with pytest.raises(ValueError, match=r"Group is zarr_format: 2"):
            write_image(data, self.group, axes="zyx", fmt=CurrentFormat())

        with pytest.raises(ValueError, match=r"Group is zarr_format: 3"):
            write_multiscale([data], self.group_v3, fmt=FormatV04())

        with pytest.raises(ValueError, match=r"Group is zarr_format: 3"):
            write_plate_metadata(self.group_v3, ["A"], ["1"], ["A/1"], fmt=FormatV04())

        with pytest.raises(ValueError, match=r"Group is zarr_format: 2"):
            write_well_metadata(self.group, [{"path": "0"}], fmt=CurrentFormat())

    @pytest.mark.parametrize("zarr_format", [2, 3])
    @pytest.mark.parametrize("array_constructor", [np.array, da.from_array])
    def test_write_image_current(self, array_constructor, zarr_format):
        shape = (64, 64, 64)
        data = self.create_data(shape)
        data = array_constructor(data)

        if zarr_format == 2:
            group = self.group
            grp_path = self.path / "test"
        else:
            group = self.group_v3
            grp_path = self.path_v3 / "test"

        write_image(data, group, axes="zyx")
        reader = Reader(parse_url(f"{grp_path}"))

        # manually check this is zarr v2 or v3
        if zarr_format == 2:
            json_text = (grp_path / ".zattrs").read_text(encoding="utf-8")
            attrs_json = json.loads(json_text)
        else:
            json_text = (grp_path / "zarr.json").read_text(encoding="utf-8")
            attrs_json = json.loads(json_text).get("attributes", {}).get("ome", {})
        assert "multiscales" in attrs_json

        image_node = next(iter(reader()))
        for transfs in image_node.metadata["coordinateTransformations"]:
            assert len(transfs) == 1
            assert transfs[0]["type"] == "scale"
            assert len(transfs[0]["scale"]) == len(shape)
            # Scaler only downsamples x and y. z scale will be 1
            assert transfs[0]["scale"][0] == 1
            for value in transfs[0]["scale"]:
                assert value >= 1

    @pytest.mark.parametrize("read_from_zarr", [True, False])
    @pytest.mark.parametrize("compute", [True, False])
    @pytest.mark.parametrize("zarr_format", [2, 3])
    def test_write_image_dask(self, read_from_zarr, compute, zarr_format):
        if zarr_format == 2:
            grp_path = self.path / "test"
            fmt = FormatV04()
            zarr_attrs = ".zattrs"
            zarr_array = ".zarray"
            group = self.group
        else:
            grp_path = self.path_v3 / "test"
            fmt = CurrentFormat()
            zarr_attrs = "zarr.json"
            zarr_array = "zarr.json"
            group = self.group_v3

        # Size 100 tests resize shapes: https://github.com/ome/ome-zarr-py/issues/219
        shape = (128, 200, 200)
        data = self.create_data(shape)
        data_delayed = da.from_array(data)
        chunks = (32, 32)
        # same NAME needed for exact zarr_attrs match below
        # (otherwise group.name is used)
        NAME = "test_write_image_dask"
        opts = {"chunks": chunks}
        if read_from_zarr:
            # write to zarr and re-read as dask...
            path = f"{grp_path}/temp/"
            store = parse_url(path, mode="w", fmt=fmt).store
            # store and group will be zarr v2 or v3 depending on fmt
            temp_group = zarr.group(store=store).create_group("to_dask")
            assert temp_group.info._zarr_format == zarr_format
            write_image(
                data_delayed,
                temp_group,
                axes="zyx",
                storage_options=opts,
                name=NAME,
            )
            print("PATH", f"{grp_path}/temp/to_dask")
            loc = ZarrLocation(f"{grp_path}/temp/to_dask")

            reader = Reader(loc)()
            nodes = list(reader)
            data_delayed = nodes[0].load(Multiscales).array(resolution="0")
            # check that the data is the same
            assert np.allclose(data, data_delayed[...].compute())

        assert group.info._zarr_format == zarr_format
        dask_delayed_jobs = write_image(
            data_delayed,
            group,
            axes="zyx",
            storage_options={"chunks": chunks},
            compute=compute,
            name=NAME,
        )

        assert not compute == len(dask_delayed_jobs)

        if not compute:
            # can be configured to use a Local or Slurm cluster
            # before persisting the jobs
            dask_delayed_jobs = persist(*dask_delayed_jobs)

        # check the data written to zarr v2 or v3 group
        reader = Reader(parse_url(f"{grp_path}"))
        image_node = next(iter(reader()))
        first_chunk = [c[0] for c in image_node.data[0].chunks]
        assert tuple(first_chunk) == _retuple(chunks, image_node.data[0].shape)
        for level, transfs in enumerate(
            image_node.metadata["coordinateTransformations"]
        ):
            assert len(transfs) == 1
            assert transfs[0]["type"] == "scale"
            assert len(transfs[0]["scale"]) == len(shape)
            # Scaler only downsamples x and y. z scale will be 1
            assert transfs[0]["scale"][0] == 1
            for value in transfs[0]["scale"]:
                assert value >= 1
            if read_from_zarr and level < 3:
                # if shape smaller than chunk, dask writer uses chunk == shape
                # so we only compare larger resolutions
                assert filecmp.cmp(
                    f"{grp_path}/temp/to_dask/{level}/{zarr_array}",
                    f"{grp_path}/{level}/{zarr_array}",
                    shallow=False,
                )

        if read_from_zarr:
            # exact match, including NAME
            assert filecmp.cmp(
                f"{grp_path}/temp/to_dask/{zarr_attrs}",
                f"{grp_path}/{zarr_attrs}",
                shallow=False,
            )

    def test_write_image_scalar_chunks(self):
        """
        Make sure a scalar chunks value is applied to all dimensions,
        matching the behaviour of zarr-python.
        """
        shape = (64, 64, 64)
        data = np.array(self.create_data(shape))
        write_image(
            image=data, group=self.group, axes="xyz", storage_options={"chunks": 32}
        )
        for data in self.group.array_values():
            print(data)
            assert data.chunks == (32, 32, 32)

    @pytest.mark.parametrize(
        "format_version",
        (
            pytest.param(FormatV04, id="V04"),
            pytest.param(FormatV05, id="V05"),
        ),
    )
    @pytest.mark.parametrize("array_constructor", [np.array, da.from_array])
    def test_write_image_compressed(self, array_constructor, format_version):
        shape = (64, 64, 64)
        data = self.create_data(shape)
        data = array_constructor(data)
        path = self.path / "test_write_image_compressed"
        store = parse_url(path, mode="w", fmt=format_version()).store
        root = zarr.group(store=store)
        CNAME = "lz4"
        LEVEL = 4
        if format_version().zarr_format == 3:
            compressor = BloscCodec(cname=CNAME, clevel=LEVEL, shuffle="shuffle")
            assert isinstance(compressor, BytesBytesCodec)
            if isinstance(data, da.Array):
                # skip test - can't get this to pass. Fails with:
                # ValueError: compressor cannot be used for arrays with zarr_format 3.
                # Use bytes-to-bytes codecs instead.
                pytest.skip("storage_options['compressor'] fails in da.to_zarr()")
        else:
            compressor = Blosc(cname=CNAME, clevel=LEVEL, shuffle=Blosc.SHUFFLE)

        write_image(
            data,
            root,
            axes="zyx",
            storage_options={"compressor": compressor},
        )
        group = zarr.open(f"{path}")
        for ds in ["0", "1"]:
            assert len(group[ds].info._compressors) > 0
            comp = group[ds].info._compressors[0]
            if format_version().zarr_format == 3:
                print("comp", comp.to_dict())
                # {'configuration': {'checksum': False, 'level': 0}, 'name': 'zstd'}
                assert comp.to_dict() == {
                    "name": "blosc",
                    "configuration": {
                        "typesize": 1,
                        "cname": CNAME,
                        "clevel": LEVEL,
                        "shuffle": "shuffle",
                        "blocksize": 0,
                    },
                }
            else:
                print("comp", comp.get_config())
                assert comp.get_config() == {
                    "id": "blosc",
                    "cname": CNAME,
                    "clevel": LEVEL,
                    "shuffle": Blosc.SHUFFLE,
                    "blocksize": 0,
                }

    @pytest.mark.parametrize(
        "format_version",
        (
            pytest.param(FormatV04, id="V04"),
            pytest.param(FormatV05, id="V05"),
        ),
    )
    @pytest.mark.parametrize("array_constructor", [np.array, da.from_array])
    def test_default_compression(self, array_constructor, format_version):
        """Test that the default compression is not None.

        We make an array of zeros which should compress trivially easily,
        write out the chunks, and check that they are smaller than the raw
        data.
        """
        arr_np = np.zeros((2, 50, 200, 400), dtype=np.uint8)
        # avoid empty chunks so they are guaranteed to be written out to disk
        arr_np[0, 0, 0, 0] = 1
        # 4MB chunks, trivially compressible
        arr = array_constructor(arr_np)
        # tempdir = TemporaryDirectory(suffix=".ome.zarr")
        # self.path = pathlib.Path(tmpdir.mkdir("data"))
        path = self.path / "test_default_compression"
        store = parse_url(path, mode="w", fmt=format_version()).store
        root = zarr.group(store=store)
        assert root.info._zarr_format == format_version().zarr_format
        # no compressor options, we are checking default
        write_image(
            arr, group=root, axes="tzyx", storage_options=dict(chunks=(1, 100, 100))
        )

        # check chunk: multiscale level 0, 4D chunk at (0, 0, 0, 0)
        c = ""
        for ds in ["0", "1"]:
            if format_version().zarr_format == 3:
                assert (path / "zarr.json").exists()
                assert (path / ds / "zarr.json").exists()
                c = "c/"
                json_text = (path / ds / "zarr.json").read_text(encoding="utf-8")
                arr_json = json.loads(json_text)
                assert arr_json["codecs"][0]["name"] == "bytes"
                assert arr_json["codecs"][1] == {
                    "name": "zstd",
                    "configuration": {"level": 0, "checksum": False},
                }
            else:
                assert (path / ".zattrs").exists()
                json_text = (path / ds / ".zarray").read_text(encoding="utf-8")
                arr_json = json.loads(json_text)
                assert arr_json["compressor"] == {
                    "blocksize": 0,
                    "clevel": 5,
                    "cname": "zstd",
                    "id": "blosc",
                    "shuffle": 1,
                }

        chunk_size = (path / f"0/{c}0/0/0/0").stat().st_size
        assert chunk_size < 4e6

    @pytest.mark.parametrize(
        "format_version",
        (
            pytest.param(FormatV04, id="V04"),
            pytest.param(FormatV05, id="V05"),
        ),
    )
    def test_validate_coordinate_transforms(self, format_version):
        fmt = format_version()

        transformations = [
            [{"type": "scale", "scale": (1, 1)}],
            [{"type": "scale", "scale": (0.5, 0.5)}],
        ]
        fmt.validate_coordinate_transformations(2, 2, transformations)

        with pytest.raises(ValueError):
            # transformations different length than levels
            fmt.validate_coordinate_transformations(2, 1, transformations)

        with pytest.raises(ValueError):
            transf = [[{"type": "scale", "scale": ("1", 1)}]]
            fmt.validate_coordinate_transformations(2, 1, transf)

        with pytest.raises(ValueError):
            transf = [[{"type": "foo", "scale": (1, 1)}]]
            fmt.validate_coordinate_transformations(2, 1, transf)

        with pytest.raises(ValueError):
            # scale list of floats different length from 3
            transf = [[{"type": "scale", "scale": (1, 1)}]]
            fmt.validate_coordinate_transformations(3, 1, transf)

        translate = [{"type": "translation", "translation": (1, 1)}]
        scale_then_trans = [transf + translate for transf in transformations]

        fmt.validate_coordinate_transformations(2, 2, scale_then_trans)

        trans_then_scale = [translate + transf for transf in transformations]
        with pytest.raises(ValueError):
            # scale must come first
            fmt.validate_coordinate_transformations(2, 2, trans_then_scale)

        with pytest.raises(ValueError):
            scale_then_trans2 = [transf + translate for transf in scale_then_trans]
            # more than 1 transformation
            fmt.validate_coordinate_transformations(2, 2, scale_then_trans2)

    def test_dim_names(self):
        v03 = FormatV03()

        # v0.3 MUST specify axes for 3D or 4D data
        with pytest.raises(ValueError):
            _get_valid_axes(3, axes=None, fmt=v03)

        # ndims must match axes length
        with pytest.raises(ValueError):
            _get_valid_axes(3, axes="yx", fmt=v03)

        # axes must be ordered tczyx
        with pytest.raises(ValueError):
            _get_valid_axes(3, axes="yxt", fmt=v03)
        with pytest.raises(ValueError):
            _get_valid_axes(2, axes=["x", "y"], fmt=v03)
        with pytest.raises(ValueError):
            _get_valid_axes(5, axes="xyzct", fmt=v03)

        # valid axes - no change, converted to list
        assert _get_valid_axes(2, axes=["y", "x"], fmt=v03) == ["y", "x"]
        assert _get_valid_axes(5, axes="tczyx", fmt=v03) == [
            "t",
            "c",
            "z",
            "y",
            "x",
        ]

        # if 2D or 5D, axes can be assigned automatically
        assert _get_valid_axes(2, axes=None, fmt=v03) == ["y", "x"]
        assert _get_valid_axes(5, axes=None, fmt=v03) == ["t", "c", "z", "y", "x"]

        # for v0.1 or v0.2, axes should be None
        assert _get_valid_axes(2, axes=["y", "x"], fmt=FormatV01()) is None
        assert _get_valid_axes(2, axes=["y", "x"], fmt=FormatV02()) is None

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
            _get_valid_axes(2, axes=[{"name": "y"}, {}], fmt=v04)

        all_dims = [
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ]

        # auto axes for 2D, 5D, converted to dict for v0.4
        assert _get_valid_axes(2, axes=None, fmt=v04) == all_dims[-2:]
        assert _get_valid_axes(5, axes=None, fmt=v04) == all_dims

        # convert from list or string
        assert _get_valid_axes(3, axes=["z", "y", "x"], fmt=v04) == all_dims[-3:]
        assert _get_valid_axes(4, axes="czyx", fmt=v04) == all_dims[-4:]

        # invalid based on ordering of types
        with pytest.raises(ValueError):
            assert _get_valid_axes(3, axes=["y", "c", "x"], fmt=v04)
        with pytest.raises(ValueError):
            assert _get_valid_axes(4, axes="ctyx", fmt=v04)

        # custom types
        assert _get_valid_axes(3, axes=["foo", "y", "x"], fmt=v04) == [
            {"name": "foo"},
            all_dims[-2],
            all_dims[-1],
        ]

        # space types can be in ANY order
        assert _get_valid_axes(3, axes=["x", "z", "y"], fmt=v04) == [
            all_dims[-1],
            all_dims[-3],
            all_dims[-2],
        ]

        # Not allowed multiple custom types
        with pytest.raises(ValueError):
            _get_valid_axes(4, axes=["foo", "bar", "y", "x"], fmt=v04)

        # unconventional naming is allowed
        strange_axes = [
            {"name": "duration", "type": "time"},
            {"name": "rotation", "type": "angle"},
            {"name": "dz", "type": "space"},
            {"name": "WIDTH", "type": "space"},
        ]
        assert _get_valid_axes(4, axes=strange_axes, fmt=v04) == strange_axes

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
        # create zarr v2 group...
        self.store = parse_url(self.path, mode="w", fmt=FormatV04()).store
        self.root = zarr.group(store=self.store)

        # let's create zarr v3 group too...
        self.path_v3 = self.path / "v3"
        store_v3 = parse_url(self.path_v3, mode="w").store
        self.root_v3 = zarr.group(store=store_v3)

    @pytest.mark.parametrize("fmt", (FormatV04(), FormatV05()))
    def test_multi_levels_transformations(self, fmt):
        datasets = []
        for level, transf in enumerate(TRANSFORMATIONS):
            datasets.append({"path": str(level), "coordinateTransformations": transf})
        if fmt.version == "0.5":
            group = self.root_v3
        else:
            group = self.root
        write_multiscales_metadata(group, datasets, axes="tczyx")
        # we want to be sure this is zarr v2 / v3
        attrs = group.attrs
        if fmt.version == "0.5":
            attrs = attrs.get("ome")
            assert "version" in attrs
            json_text = (self.path_v3 / "zarr.json").read_text(encoding="utf-8")
            attrs_json = json.loads(json_text).get("attributes", {}).get("ome", {})
        else:
            json_text = (self.path / ".zattrs").read_text(encoding="utf-8")
            attrs_json = json.loads(json_text)
            assert "version" in attrs["multiscales"][0]
        assert "multiscales" in attrs_json
        assert "multiscales" in attrs
        assert attrs["multiscales"][0]["datasets"] == datasets

    @pytest.mark.parametrize("fmt", (FormatV01(), FormatV02(), FormatV03()))
    def test_version(self, fmt):
        write_multiscales_metadata(self.root, [{"path": "0"}], fmt=fmt)
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
    def test_axes_V03(self, axes):
        write_multiscales_metadata(
            self.root, [{"path": "0"}], fmt=FormatV03(), axes=axes
        )
        assert "multiscales" in self.root.attrs
        # for v0.3, axes is a list of names
        assert self.root.attrs["multiscales"][0]["axes"] == axes
        with pytest.raises(ValueError):
            # for v0.4 and above, paths no-longer supported (need dataset dicts)
            write_multiscales_metadata(self.root, ["0"], axes=axes, fmt=FormatV04())

    @pytest.mark.parametrize("fmt", (FormatV01(), FormatV02()))
    def test_axes_ignored(self, fmt):
        write_multiscales_metadata(
            self.root, [{"path": "0"}], fmt=fmt, axes=["t", "c", "z", "y", "x"]
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

    @pytest.mark.parametrize("datasets", ([], None, "0", ["0"], [{"key": 1}]))
    def test_invalid_datasets(self, datasets):
        with pytest.raises(ValueError):
            write_multiscales_metadata(
                self.root, datasets, axes=["t", "c", "z", "y", "x"], fmt=FormatV04()
            )

    @pytest.mark.parametrize(
        "coordinateTransformations",
        (
            [{"type": "scale", "scale": [1, 1]}],
            [
                {"type": "scale", "scale": [1, 1]},
                {"type": "translation", "translation": [0, 0]},
            ],
        ),
    )
    def test_valid_transformations(self, coordinateTransformations):
        axes = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
        datasets = [
            {
                "path": "0",
                "coordinateTransformations": coordinateTransformations,
            }
        ]
        write_multiscales_metadata(self.root, datasets, axes=axes, fmt=FormatV04())
        assert "multiscales" in self.root.attrs
        assert self.root.attrs["multiscales"][0]["axes"] == axes
        assert self.root.attrs["multiscales"][0]["datasets"] == datasets

    @pytest.mark.parametrize(
        "coordinateTransformations",
        (
            [],
            None,
            [{"type": "scale"}],
            [{"scale": [1, 1]}],
            [{"type": "scale", "scale": ["1", 1]}],
            [{"type": "scale", "scale": [1, 1, 1]}],
            [{"type": "scale", "scale": [1, 1]}, {"type": "scale", "scale": [1, 1]}],
            [
                {"type": "scale", "scale": [1, 1]},
                {"type": "translation", "translation": ["0", 0]},
            ],
            [
                {"type": "translation", "translation": [0, 0]},
            ],
            [
                {"type": "scale", "scale": [1, 1]},
                {"type": "translation", "translation": [0, 0, 0]},
            ],
            [
                {"type": "translation", "translation": [0, 0]},
                {"type": "scale", "scale": [1, 1]},
            ],
            [
                {"type": "scale", "scale": [1, 1]},
                {"type": "translation", "translation": [0, 0]},
                {"type": "translation", "translation": [1, 0]},
            ],
            [
                {"type": "scale", "scale": [1, 1]},
                {"translation": [0, 0]},
            ],
            [
                {"type": "scale", "scale": [1, 1]},
                {"type": "translation", "translate": [0, 0]},
            ],
        ),
    )
    def test_invalid_transformations(self, coordinateTransformations):
        axes = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
        datasets = [
            {"path": "0", "coordinateTransformations": coordinateTransformations}
        ]
        with pytest.raises(ValueError):
            write_multiscales_metadata(self.root, datasets, axes=axes, fmt=FormatV04())

    @pytest.mark.parametrize(
        "metadata",
        [
            {
                "channels": [
                    {
                        "color": "FF0000",
                        "window": {"start": 0, "end": 255, "min": 0, "max": 255},
                    }
                ]
            },
            {"channels": [{"color": "FF0000"}]},
            {"channels": [{"color": "FF000"}]},  # test wrong metadata
            {"channels": [{"window": []}]},  # test wrong metadata
            {
                "channels": [  # test wrong metadata
                    {"color": "FF0000", "window": {"start": 0, "end": 255, "min": 0}},
                ]
            },
            None,
        ],
    )
    def test_omero_metadata(self, metadata: dict[str, Any] | None):
        datasets = []
        for level, transf in enumerate(TRANSFORMATIONS):
            datasets.append({"path": str(level), "coordinateTransformations": transf})
        if metadata is None:
            with pytest.raises(
                KeyError, match="If `'omero'` is present, value cannot be `None`."
            ):
                write_multiscales_metadata(
                    self.root,
                    datasets,
                    axes="tczyx",
                    metadata={"omero": metadata},
                )
        else:
            window_metadata = (
                metadata["channels"][0].get("window")
                if "window" in metadata["channels"][0]
                else None
            )
            color_metadata = (
                metadata["channels"][0].get("color")
                if "color" in metadata["channels"][0]
                else None
            )
            if window_metadata is not None and len(window_metadata) < 4:
                if isinstance(window_metadata, dict):
                    with pytest.raises(KeyError, match=".*`'window'`.*"):
                        write_multiscales_metadata(
                            self.root,
                            datasets,
                            axes="tczyx",
                            metadata={"omero": metadata},
                            fmt=FormatV04(),
                        )
                elif isinstance(window_metadata, list):
                    with pytest.raises(TypeError, match=".*`'window'`.*"):
                        write_multiscales_metadata(
                            self.root,
                            datasets,
                            axes="tczyx",
                            metadata={"omero": metadata},
                            fmt=FormatV04(),
                        )
            elif color_metadata is not None and len(color_metadata) != 6:
                with pytest.raises(TypeError, match=".*`'color'`.*"):
                    write_multiscales_metadata(
                        self.root,
                        datasets,
                        axes="tczyx",
                        metadata={"omero": metadata},
                    )
            else:
                write_multiscales_metadata(
                    self.root,
                    datasets,
                    axes="tczyx",
                    metadata={"omero": metadata},
                )


class TestPlateMetadata:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = pathlib.Path(tmpdir.mkdir("data"))
        # create zarr v2 group...
        self.store = parse_url(self.path, mode="w", fmt=FormatV04()).store
        self.root = zarr.group(store=self.store)
        # create zarr v3 group...
        self.path_v3 = self.path / "v3"
        store_v3 = parse_url(self.path_v3, mode="w").store
        self.root_v3 = zarr.group(store=store_v3)

    @pytest.mark.parametrize("fmt", (FormatV04(), FormatV05()))
    def test_minimal_plate(self, fmt):
        if fmt.version == "0.4":
            group = self.root
        else:
            group = self.root_v3
        write_plate_metadata(group, ["A"], ["1"], ["A/1"])
        attrs = group.attrs
        if fmt.version != "0.4":
            attrs = attrs["ome"]
            assert attrs["version"] == fmt.version
        else:
            assert attrs["plate"]["version"] == fmt.version

        assert "plate" in attrs
        assert attrs["plate"]["columns"] == [{"name": "1"}]
        assert attrs["plate"]["rows"] == [{"name": "A"}]
        assert attrs["plate"]["wells"] == [
            {"path": "A/1", "rowIndex": 0, "columnIndex": 0}
        ]
        assert "name" not in attrs["plate"]
        assert "field_count" not in attrs["plate"]
        assert "acquisitions" not in attrs["plate"]

    @pytest.mark.parametrize("fmt", (FormatV04(), FormatV05()))
    def test_12wells_plate(self, fmt):
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
        if fmt.version == "0.4":
            group = self.root
        else:
            group = self.root_v3
        write_plate_metadata(group, rows, cols, wells)
        attrs = group.attrs
        if fmt.version != "0.4":
            attrs = attrs["ome"]

        assert "plate" in attrs
        assert attrs["plate"]["columns"] == [
            {"name": "1"},
            {"name": "2"},
            {"name": "3"},
        ]
        assert attrs["plate"]["rows"] == [
            {"name": "A"},
            {"name": "B"},
            {"name": "C"},
            {"name": "D"},
        ]
        assert attrs["plate"]["wells"] == [
            {"path": "A/1", "rowIndex": 0, "columnIndex": 0},
            {"path": "A/2", "rowIndex": 0, "columnIndex": 1},
            {"path": "A/3", "rowIndex": 0, "columnIndex": 2},
            {"path": "B/1", "rowIndex": 1, "columnIndex": 0},
            {"path": "B/2", "rowIndex": 1, "columnIndex": 1},
            {"path": "B/3", "rowIndex": 1, "columnIndex": 2},
            {"path": "C/1", "rowIndex": 2, "columnIndex": 0},
            {"path": "C/2", "rowIndex": 2, "columnIndex": 1},
            {"path": "C/3", "rowIndex": 2, "columnIndex": 2},
            {"path": "D/1", "rowIndex": 3, "columnIndex": 0},
            {"path": "D/2", "rowIndex": 3, "columnIndex": 1},
            {"path": "D/3", "rowIndex": 3, "columnIndex": 2},
        ]
        assert "name" not in attrs["plate"]
        assert "field_count" not in attrs["plate"]
        assert "acquisitions" not in attrs["plate"]

    @pytest.mark.parametrize("fmt", (FormatV04(), FormatV05()))
    def test_sparse_plate(self, fmt):
        rows = ["A", "B", "C", "D", "E"]
        cols = ["1", "2", "3", "4", "5"]
        wells = [
            "B/2",
            "E/5",
        ]
        if fmt.version == "0.4":
            group = self.root
        else:
            group = self.root_v3
        write_plate_metadata(group, rows, cols, wells)
        attrs = group.attrs
        if fmt.version != "0.4":
            attrs = attrs["ome"]
        assert "plate" in attrs
        assert attrs["plate"]["columns"] == [
            {"name": "1"},
            {"name": "2"},
            {"name": "3"},
            {"name": "4"},
            {"name": "5"},
        ]
        assert attrs["plate"]["rows"] == [
            {"name": "A"},
            {"name": "B"},
            {"name": "C"},
            {"name": "D"},
            {"name": "E"},
        ]
        assert attrs["plate"]["wells"] == [
            {"path": "B/2", "rowIndex": 1, "columnIndex": 1},
            {"path": "E/5", "rowIndex": 4, "columnIndex": 4},
        ]
        assert "name" not in attrs["plate"]
        assert "field_count" not in attrs["plate"]
        assert "acquisitions" not in attrs["plate"]

    @pytest.mark.parametrize("fmt", (FormatV01(), FormatV02(), FormatV03()))
    def test_legacy_wells(self, fmt):
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
        # We don't need to test v04 and v05 for all tests since
        # the metadata is the same
        write_plate_metadata(self.root_v3, ["A"], ["1"], ["A/1"], name="test")
        attrs = self.root_v3.attrs["ome"]
        assert "plate" in attrs
        assert attrs["plate"]["columns"] == [{"name": "1"}]
        assert attrs["plate"]["name"] == "test"
        assert attrs["plate"]["rows"] == [{"name": "A"}]
        assert attrs["version"] == FormatV05().version
        assert attrs["plate"]["wells"] == [
            {"path": "A/1", "rowIndex": 0, "columnIndex": 0}
        ]
        assert "field_count" not in attrs["plate"]
        assert "acquisitions" not in attrs["plate"]

    def test_field_count(self):
        write_plate_metadata(
            self.root, ["A"], ["1"], ["A/1"], field_count=10, fmt=FormatV04()
        )
        assert "plate" in self.root.attrs
        assert self.root.attrs["plate"]["columns"] == [{"name": "1"}]
        assert self.root.attrs["plate"]["field_count"] == 10
        assert self.root.attrs["plate"]["rows"] == [{"name": "A"}]
        assert self.root.attrs["plate"]["version"] == FormatV04().version
        assert self.root.attrs["plate"]["wells"] == [
            {"path": "A/1", "rowIndex": 0, "columnIndex": 0}
        ]
        assert "name" not in self.root.attrs["plate"]
        assert "acquisitions" not in self.root.attrs["plate"]

    def test_acquisitions_minimal(self):
        a = [{"id": 1}, {"id": 2}, {"id": 3}]
        write_plate_metadata(
            self.root, ["A"], ["1"], ["A/1"], acquisitions=a, fmt=FormatV04()
        )
        assert "plate" in self.root.attrs
        assert self.root.attrs["plate"]["acquisitions"] == a
        assert self.root.attrs["plate"]["columns"] == [{"name": "1"}]
        assert self.root.attrs["plate"]["rows"] == [{"name": "A"}]
        assert self.root.attrs["plate"]["version"] == FormatV04().version
        assert self.root.attrs["plate"]["wells"] == [
            {"path": "A/1", "rowIndex": 0, "columnIndex": 0}
        ]
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
        write_plate_metadata(
            self.root, ["A"], ["1"], ["A/1"], acquisitions=a, fmt=FormatV04()
        )
        assert "plate" in self.root.attrs
        assert self.root.attrs["plate"]["acquisitions"] == a
        assert self.root.attrs["plate"]["columns"] == [{"name": "1"}]
        assert self.root.attrs["plate"]["rows"] == [{"name": "A"}]
        assert self.root.attrs["plate"]["version"] == FormatV04().version
        assert self.root.attrs["plate"]["wells"] == [
            {"path": "A/1", "rowIndex": 0, "columnIndex": 0}
        ]
        assert "name" not in self.root.attrs["plate"]
        assert "field_count" not in self.root.attrs["plate"]

    @pytest.mark.parametrize(
        "acquisitions",
        (
            [0, 1],
            [{"name": "0"}, {"name": "1"}],
            [{"id": "0"}, {"id": "1"}],
        ),
    )
    def test_invalid_acquisition_keys(self, acquisitions):
        with pytest.raises(ValueError):
            write_plate_metadata(
                self.root_v3, ["A"], ["1"], ["A/1"], acquisitions=acquisitions
            )

    def test_unspecified_acquisition_keys(self):
        a = [{"id": 0, "unspecified_key": "0"}]
        write_plate_metadata(self.root, ["A"], ["1"], ["A/1"], acquisitions=a)
        assert "plate" in self.root.attrs
        assert self.root.attrs["plate"]["acquisitions"] == a

    @pytest.mark.parametrize(
        "wells",
        (None, [], [1]),
    )
    def test_invalid_well_list(self, wells):
        with pytest.raises(ValueError):
            write_plate_metadata(self.root, ["A"], ["1"], wells)

    @pytest.mark.parametrize(
        "wells",
        (
            # Missing required keys
            [{"id": "test"}],
            [{"path": "A/1"}],
            [{"path": "A/1", "rowIndex": 0}],
            [{"path": "A/1", "columnIndex": 0}],
            [{"rowIndex": 0, "columnIndex": 0}],
            # Invalid paths
            [{"path": 0, "rowIndex": 0, "columnIndex": 0}],
            [{"path": None, "rowIndex": 0, "columnIndex": 0}],
            [{"path": "plate/A/1", "rowIndex": 0, "columnIndex": 0}],
            [{"path": "plate/A1", "rowIndex": 0, "columnIndex": 0}],
            [{"path": "A/1/0", "rowIndex": 0, "columnIndex": 0}],
            [{"path": "A1", "rowIndex": 0, "columnIndex": 0}],
            [{"path": "0", "rowIndex": 0, "columnIndex": 0}],
            # Invalid row/column indices
            [{"path": "A/1", "rowIndex": "0", "columnIndex": 0}],
            [{"path": "A/1", "rowIndex": 0, "columnIndex": "0"}],
            # Undefined rows/columns
            [{"path": "C/1", "rowIndex": 2, "columnIndex": 0}],
            [{"path": "A/3", "rowIndex": 0, "columnIndex": 2}],
            # Mismatching indices
            [{"path": "A/1", "rowIndex": 0, "columnIndex": 1}],
            [{"path": "A/1", "rowIndex": 1, "columnIndex": 0}],
        ),
    )
    def test_invalid_well_keys(self, wells):
        with pytest.raises(ValueError):
            write_plate_metadata(self.root, ["A"], ["1"], wells, fmt=FormatV04())

    @pytest.mark.parametrize("fmt", (FormatV01(), FormatV02(), FormatV03()))
    def test_legacy_unspecified_well_keys(self, fmt):
        wells = [
            {"path": "A/1", "unspecified_key": "alpha"},
            {"path": "A/2", "unspecified_key": "beta"},
            {"path": "B/1", "unspecified_key": "gamma"},
        ]
        write_plate_metadata(self.root, ["A", "B"], ["1", "2"], wells, fmt=fmt)
        assert "plate" in self.root.attrs
        assert self.root.attrs["plate"]["columns"] == [{"name": "1"}, {"name": "2"}]
        assert self.root.attrs["plate"]["rows"] == [{"name": "A"}, {"name": "B"}]
        assert self.root.attrs["plate"]["version"] == fmt.version
        assert self.root.attrs["plate"]["wells"] == wells

    def test_unspecified_well_keys(self):
        wells = [
            {
                "path": "A/1",
                "rowIndex": 0,
                "columnIndex": 0,
                "unspecified_key": "alpha",
            },
            {"path": "A/2", "rowIndex": 0, "columnIndex": 1, "unspecified_key": "beta"},
            {
                "path": "B/1",
                "rowIndex": 1,
                "columnIndex": 0,
                "unspecified_key": "gamma",
            },
        ]
        write_plate_metadata(self.root, ["A", "B"], ["1", "2"], wells, fmt=FormatV04())
        assert "plate" in self.root.attrs
        assert self.root.attrs["plate"]["columns"] == [{"name": "1"}, {"name": "2"}]
        assert self.root.attrs["plate"]["rows"] == [{"name": "A"}, {"name": "B"}]
        assert self.root.attrs["plate"]["version"] == FormatV04().version
        assert self.root.attrs["plate"]["wells"] == wells

    def test_missing_well_keys(self):
        wells = [
            {"path": "A/1"},
            {"path": "A/2"},
            {"path": "B/1"},
        ]
        with pytest.raises(ValueError):
            write_plate_metadata(
                self.root, ["A", "B"], ["1", "2"], wells, fmt=FormatV04()
            )

    def test_well_not_in_rows(self):
        wells = ["A/1", "B/1", "C/1"]
        with pytest.raises(ValueError):
            write_plate_metadata(
                self.root, ["A", "B"], ["1", "2"], wells, fmt=FormatV04()
            )

    def test_well_not_in_columns(self):
        wells = ["A/1", "A/2", "A/3"]
        with pytest.raises(ValueError):
            write_plate_metadata(
                self.root, ["A", "B"], ["1", "2"], wells, fmt=FormatV04()
            )

    @pytest.mark.parametrize("rows", (["A", "B", "B"], ["A", "&"]))
    def test_invalid_rows(self, rows):
        with pytest.raises(ValueError):
            write_plate_metadata(self.root, rows, ["1"], ["A/1"], fmt=FormatV04())

    @pytest.mark.parametrize("columns", (["1", "2", "2"], ["1", "&"]))
    def test_invalid_columns(self, columns):
        with pytest.raises(ValueError):
            write_plate_metadata(self.root, ["A"], columns, ["A/1"], fmt=FormatV04())


class TestWellMetadata:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = pathlib.Path(tmpdir.mkdir("data"))
        # create zarr v2 group...
        self.store = parse_url(self.path, mode="w", fmt=FormatV04()).store
        self.root = zarr.group(store=self.store)

        # create zarr v3 group too...
        self.path_v3 = self.path / "v3"
        store_v3 = parse_url(self.path_v3, mode="w").store
        self.root_v3 = zarr.group(store=store_v3)

    @pytest.mark.parametrize("fmt", (FormatV04(), FormatV05()))
    @pytest.mark.parametrize("images", (["0"], [{"path": "0"}]))
    def test_minimal_well(self, images, fmt):
        if fmt.version == "0.5":
            group = self.root_v3
        else:
            group = self.root
        write_well_metadata(group, images)
        # we want to be sure this is zarr v2 / v3, so we load json manually too
        attrs = group.attrs
        if fmt.version == "0.5":
            attrs = attrs.get("ome")
            assert attrs["version"] == fmt.version
            json_text = (self.path_v3 / "zarr.json").read_text(encoding="utf-8")
            attrs_json = json.loads(json_text).get("attributes", {}).get("ome", {})
        else:
            json_text = (self.path / ".zattrs").read_text(encoding="utf-8")
            attrs_json = json.loads(json_text)
            assert attrs["well"]["version"] == fmt.version

        assert "well" in attrs_json
        assert attrs["well"]["images"] == [{"path": "0"}]

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
        write_well_metadata(self.root_v3, images)
        assert "well" in self.root_v3.attrs.get("ome", {})
        assert self.root_v3.attrs["ome"]["well"]["images"] == [
            {"path": "0"},
            {"path": "1"},
            {"path": "2"},
        ]
        assert self.root_v3.attrs["ome"]["version"] == FormatV05().version

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
        assert self.root.attrs["well"]["version"] == FormatV04().version

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
        assert self.root.attrs["well"]["version"] == FormatV04().version


class TestLabelWriter:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = pathlib.Path(tmpdir.mkdir("data"))
        # create zarr v2 group...
        self.store = parse_url(self.path, mode="w", fmt=FormatV04()).store
        self.root = zarr.group(store=self.store)
        # create zarr v3 group...
        self.path_v3 = self.path / "v3"
        store_v3 = parse_url(self.path_v3, mode="w").store
        self.root_v3 = zarr.group(store=store_v3)

    def create_image_data(self, group, shape, scaler, fmt, axes, transformations):
        rng = np.random.default_rng(0)
        data = rng.poisson(10, size=shape).astype(np.uint8)
        write_image(
            image=data,
            group=group,
            scaler=scaler,
            fmt=fmt,
            axes=axes,
            coordinate_transformations=transformations,
            storage_options=dict(chunks=(128, 128)),
        )

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

    def verify_label_data(
        self, img_path, label_name, label_data, fmt, shape, transformations
    ):
        # Verify image data
        reader = Reader(parse_url(f"{img_path}/labels/{label_name}"))
        node = next(iter(reader()))
        assert Multiscales.matches(node.zarr)
        if fmt.version in ("0.1", "0.2"):
            # v0.1 and v0.2 MUST be 5D
            assert node.data[0].ndim == 5
        else:
            assert node.data[0].shape == shape

        if fmt.version not in ("0.1", "0.2", "0.3"):
            for transf, expected in zip(
                node.metadata["coordinateTransformations"], transformations
            ):
                assert transf == expected
            assert len(node.metadata["coordinateTransformations"]) == len(node.data)
        assert np.allclose(label_data, node.data[0][...].compute())

        # Verify label metadata
        label_root = zarr.open(f"{img_path}/labels", mode="r")
        label_attrs = label_root.attrs
        if fmt.version == "0.5":
            label_attrs = label_attrs["ome"]
        assert "labels" in label_attrs
        assert label_name in label_attrs["labels"]

        label_group = zarr.open(f"{img_path}/labels/{label_name}", mode="r")
        imglabel_attrs = label_group.attrs
        if fmt.version == "0.5":
            imglabel_attrs = imglabel_attrs["ome"]
            assert imglabel_attrs["version"] == fmt.version
        else:
            assert imglabel_attrs["image-label"]["version"] == fmt.version
        assert "image-label" in imglabel_attrs

        # Verify multiscale metadata
        name = imglabel_attrs["multiscales"][0].get("name", "")
        assert label_name == name

    @pytest.mark.parametrize(
        "format_version",
        (
            pytest.param(FormatV01, id="V01"),
            pytest.param(FormatV02, id="V02"),
            pytest.param(FormatV03, id="V03"),
            pytest.param(FormatV04, id="V04"),
            pytest.param(FormatV05, id="V05"),
        ),
    )
    @pytest.mark.parametrize("array_constructor", [np.array, da.from_array])
    def test_write_labels(self, shape, scaler, format_version, array_constructor):
        fmt = format_version()
        if fmt.version == "0.5":
            img_path = self.path_v3
            group = self.root_v3
        else:
            img_path = self.path
            group = self.root

        axes = "tczyx"[-len(shape) :]
        transformations = []
        for dataset_transfs in TRANSFORMATIONS:
            transf = dataset_transfs[0]
            # e.g. slice [1, 1, z, x, y] -> [z, x, y] for 3D
            transformations.append(
                [{"type": "scale", "scale": transf["scale"][-len(shape) :]}]
            )
            if scaler is None:
                break

        # create the actual label data
        label_data = np.random.randint(0, 1000, size=shape)
        if fmt.version in ("0.1", "0.2"):
            # v0.1 and v0.2 require 5d
            expand_dims = (np.s_[None],) * (5 - len(shape))
            label_data = label_data[expand_dims]
            assert label_data.ndim == 5
        label_name = "my-labels"
        label_data = array_constructor(label_data)

        # create the root level image data
        self.create_image_data(group, shape, scaler, fmt, axes, transformations)

        write_labels(
            label_data,
            group,
            scaler=scaler,
            name=label_name,
            fmt=fmt,
            axes=axes,
            coordinate_transformations=transformations,
        )
        self.verify_label_data(
            img_path, label_name, label_data, fmt, shape, transformations
        )

    @pytest.mark.parametrize(
        "format_version",
        (
            pytest.param(FormatV01, id="V01"),
            pytest.param(FormatV02, id="V02"),
            pytest.param(FormatV03, id="V03"),
            pytest.param(FormatV04, id="V04"),
            pytest.param(FormatV05, id="V05"),
        ),
    )
    @pytest.mark.parametrize("array_constructor", [np.array, da.from_array])
    def test_write_multiscale_labels(
        self, shape, scaler, format_version, array_constructor
    ):
        fmt = format_version()
        if fmt.version == "0.5":
            img_path = self.path_v3
            group = self.root_v3
        else:
            img_path = self.path
            group = self.root
        axes = "tczyx"[-len(shape) :]
        transformations = []
        for dataset_transfs in TRANSFORMATIONS:
            transf = dataset_transfs[0]
            # e.g. slice [1, 1, z, x, y] -> [z, x, y] for 3D
            transformations.append(
                [{"type": "scale", "scale": transf["scale"][-len(shape) :]}]
            )

        # create the actual label data
        label_data = np.random.randint(0, 1000, size=shape)
        if fmt.version in ("0.1", "0.2"):
            # v0.1 and v0.2 require 5d
            expand_dims = (np.s_[None],) * (5 - len(shape))
            label_data = label_data[expand_dims]
            assert label_data.ndim == 5
        label_data = array_constructor(label_data)

        label_name = "my-labels"
        if scaler is None:
            transformations = [transformations[0]]
            labels_mip = [label_data]
        else:
            labels_mip = scaler.nearest(label_data)

        # create the root level image data
        self.create_image_data(group, shape, scaler, fmt, axes, transformations)

        write_multiscale_labels(
            labels_mip,
            group,
            name=label_name,
            fmt=fmt,
            axes=axes,
            coordinate_transformations=transformations,
        )
        self.verify_label_data(
            img_path, label_name, label_data, fmt, shape, transformations
        )

    @pytest.mark.parametrize(
        "fmt",
        (pytest.param(FormatV04(), id="V04"), pytest.param(FormatV05(), id="V05")),
    )
    @pytest.mark.parametrize("array_constructor", [np.array, da.from_array])
    def test_two_label_images(self, array_constructor, fmt):
        if fmt.version == "0.5":
            img_path = self.path_v3
            group = self.root_v3
        else:
            img_path = self.path
            group = self.root
        axes = "tczyx"
        transformations = []
        for dataset_transfs in TRANSFORMATIONS:
            transf = dataset_transfs[0]
            transformations.append([{"type": "scale", "scale": transf["scale"]}])

        # create the root level image data
        shape = (1, 2, 1, 256, 256)
        scaler = Scaler()
        self.create_image_data(
            group,
            shape,
            scaler,
            axes=axes,
            fmt=fmt,
            transformations=transformations,
        )

        label_names = ("first_labels", "second_labels")
        for label_name in label_names:
            label_data = np.random.randint(0, 1000, size=shape)
            label_data = array_constructor(label_data)
            labels_mip = scaler.nearest(label_data)

            write_multiscale_labels(
                labels_mip,
                group,
                name=label_name,
                fmt=fmt,
                axes=axes,
                coordinate_transformations=transformations,
            )
            self.verify_label_data(
                img_path, label_name, label_data, fmt, shape, transformations
            )

        # Verify label metadata
        label_root = zarr.open(f"{img_path}/labels", mode="r")
        attrs = label_root.attrs
        if fmt.version == "0.5":
            attrs = attrs["ome"]
        assert "labels" in attrs
        assert len(attrs["labels"]) == len(label_names)
        assert all(label_name in attrs["labels"] for label_name in label_names)
