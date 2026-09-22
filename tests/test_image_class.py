import dask.array as da
import numpy as np
import pytest
import zarr

from ome_zarr import OMEZarrImage, OMEZarrLabels, OMEZarrMultiscale
from ome_zarr.writer import _retuple

# (shape, axes, scale) for image data ranging from 2D to 5D
IMAGE_DIMS = [
    ((128, 128), "yx", {"y": 0.5, "x": 0.5}),
    ((2, 128, 128), "cyx", {"y": 0.5, "x": 0.5}),
    ((2, 10, 128, 128), "czyx", {"z": 1.0, "y": 0.5, "x": 0.5}),
    ((3, 2, 10, 128, 128), "tczyx", {"z": 1.0, "y": 0.5, "x": 0.5}),
]


@pytest.fixture(params=IMAGE_DIMS, ids=["2D", "3D", "4D", "5D"])
def image_dims(request):
    """(shape, axes, scale) for a range of dimensionalities, from 2D to 5D."""
    return request.param


def create_data(shape, dtype=np.uint8, mean_val=10):
    rng = np.random.default_rng(0)
    return rng.poisson(mean_val, size=shape).astype(dtype)


def test_additional_transforms(tmp_path, image_dims):
    """
    Test passing additional coordinate transformations to instances
    of OMEZarrMultiscale. This requires both the coordinate system
    and the coordinate transformation to be passed.
    """
    from ome_zarr_models.v06.coordinate_transforms import (
        CoordinateSystem,
        CoordinateSystemIdentifier,
        Sequence,
    )

    shape, axes, scale = image_dims
    data = create_data(shape)

    image = OMEZarrImage(data=data, axes=axes, scale=scale)

    # create an additional coordinate system with the same dimensionality
    # non-spatial axes (channel, time) aren't rescaled/translated in "world"
    axis_meta = {"c": ("channel", "none"), "t": ("time", "second")}
    world_scale = [1.0 if ax in axis_meta else 0.5 for ax in axes]
    world_translation = [0.0 if ax in axis_meta else 10.0 for ax in axes]

    additional_transforms = Sequence.model_validate(
        {
            "type": "sequence",
            "input": {"name": "physical"},
            "output": {"name": "world"},
            "transformations": [
                {"type": "scale", "scale": world_scale},
                {"type": "translation", "translation": world_translation},
            ],
        }
    )

    additional_cs = [
        CoordinateSystem.model_validate(
            {
                "name": "world",
                "axes": [
                    {
                        "name": ax,
                        "type": axis_meta.get(ax, ("space", "micrometer"))[0],
                        "unit": axis_meta.get(ax, ("space", "micrometer"))[1],
                    }
                    for ax in axes
                ],
            }
        )
    ]

    # this call lacks the coordinate system "world"
    # needed as output for the additional transforms
    with pytest.raises(ValueError):
        OMEZarrMultiscale.from_singlescale(
            image=image,
            scale_factors=None,
            method=None,
            coordinate_transformations=(additional_transforms,),
        )

    ms = OMEZarrMultiscale.from_singlescale(
        image=image,
        scale_factors=None,
        method=None,
        coordinate_transformations=(additional_transforms,),
        coordinate_systems=additional_cs,
        default_coordinate_system_name="physical",
    )
    ms.to_ome_zarr(
        zarr.open(tmp_path / "test_transforms.zarr", mode="w"),
        version="0.6",
        overwrite=True,
    )

    # make transform go bad
    additional_transforms = additional_transforms.model_copy(
        update={"output": CoordinateSystemIdentifier(name="nonexistent")}
    )
    with pytest.raises(ValueError):
        OMEZarrMultiscale.from_singlescale(
            image=image,
            scale_factors=None,
            method=None,
            coordinate_transformations=(additional_transforms,),
            coordinate_systems=additional_cs,
            default_coordinate_system_name="physical",
        )


@pytest.mark.parametrize("version", ("0.4", "0.5", "0.6"), ids=["V04", "V05", "V06"])
def test_image_class_versions(tmp_path, image_dims, version):
    """
    Check that the class-based interface correctly handles different OME-Zarr versions
    on read and write.
    """
    from ome_zarr_models.v06.multiscales import Multiscale as Multiscale_V06

    shape, axes, scale = image_dims
    data = create_data(shape)
    image = OMEZarrImage(data=data, axes=axes, scale=scale)
    ms = OMEZarrMultiscale.from_singlescale(
        image=image,
    )
    zarr_format = 2 if version == "0.4" else 3
    grp = zarr.open(
        tmp_path / f"test_versions_{version}.zarr",
        mode="w",
        zarr_format=zarr_format,
    )
    ms.to_ome_zarr(grp, overwrite=True, version=version)

    # open the written zarr and check the version
    out = zarr.open_group(tmp_path / f"test_versions_{version}.zarr")
    if version == "0.4":
        metadata = out.attrs["multiscales"][0]
    else:
        metadata = out.attrs.get("ome", {})

    assert metadata["version"] == version

    # The reader defaults to interpreting the read data in the latest
    # version, which is 0.6
    ms_read = OMEZarrMultiscale.from_ome_zarr(grp)
    assert isinstance(ms_read.metadata, Multiscale_V06)


def test_image_class_bad_args(tmp_path):
    """
    Check that the class-based interface raises appropriate errors
    when given bad arguments.
    """
    data = create_data((2, 128, 128))

    with pytest.raises(ValueError):
        OMEZarrImage(data=data, axes="czyx")  # more axes than data dims

    with pytest.raises(ValueError):
        # more scale values than data dims
        OMEZarrImage(
            data=data,
            axes="zyx",
            scale={"c": 1.0, "z": 0.5, "y": 0.5, "x": 0.5},
        )

    # unset axes must default to 1.0
    image = OMEZarrImage(data=data, axes="cyx", scale={"y": 0.5, "x": 0.5})
    assert image.scale["c"] == 1.0

    # less channels then dims in channel axis
    with pytest.raises(TypeError):
        OMEZarrMultiscale.from_singlescale(
            image=image,
            channel_names=["Channel 0"],
        )

    # less channel_names than channel_colors
    with pytest.raises(TypeError):
        OMEZarrMultiscale.from_singlescale(
            image=image,
            channel_names=["Channel 0", "Channel 1"],
            channel_colors=["#ff0000"],
        )

    # less channel_names than contrast limits
    with pytest.raises(TypeError):
        OMEZarrMultiscale.from_singlescale(
            image=image,
            channel_names=["Channel 0", "Channel 1"],
            contrast_limits=[(0, 255)],
        )

    multiscales = OMEZarrMultiscale.from_singlescale(
        image=image,
        scale_factors=None,
        method=None,
        channel_names=["Channel 0", "Channel 1"],
        channel_colors=[[255, 0, 0], [0, 255, 0]],
        contrast_limits=[(0, 255), (0, 255)],
    )
    assert len(multiscales.images) == 5

    multiscales.to_ome_zarr(tmp_path / "test_bad_args.zarr", version="0.5.5")


def test_multiscale_from_pyramid_sanity(image_dims):
    """
    Check that building an OMEZarrMultiscale from a pre-built list of images works.
    """

    def _shrink(shape, axes, factor):
        """Halve spatial axes `factor` times; leave channel/time axes untouched."""
        return tuple(
            max(1, s // factor) if a in ("z", "y", "x") else s for s, a in zip(shape, axes)
        )

    shape, axes, scale = image_dims
    levels = [
        OMEZarrImage(
            data=create_data(_shrink(shape, axes, 2**level)),
            axes=axes,
            scale={d: v * 2**level if d in ("z", "y", "x") else v for d, v in scale.items()},
        )
        for level in range(3)
    ]

    ms = OMEZarrMultiscale(image=levels, scale_factors=None, method=None)

    assert len(ms.images) == 3
    assert len(ms.metadata.datasets) == 3
    assert ms.images[0].data.shape == shape


def test_multiscale_from_pyramid_axes_mismatch():
    """
    Check that pyramid levels with inconsistent axes or axes_units are rejected.
    """
    level0 = OMEZarrImage(
        data=create_data((64, 64)), axes="yx", scale={"y": 1.0, "x": 1.0}
    )

    bad_axes = OMEZarrImage(
        data=create_data((32, 32, 32)),
        axes="zyx",
        scale={"z": 1.0, "y": 2.0, "x": 2.0},
    )
    with pytest.raises(ValueError, match="axes"):
        OMEZarrMultiscale.from_pyramid(image=[level0, bad_axes])

    bad_units = OMEZarrImage(
        data=create_data((32, 32)),
        axes="yx",
        scale={"y": 2.0, "x": 2.0},
        axes_units={"y": "millimeter", "x": "millimeter"},
    )
    with pytest.raises(ValueError, match="axes_units"):
        OMEZarrMultiscale.from_pyramid(image=[level0, bad_units])

def test_image_class_writer_default_scale():
    """Axes with no scale given default to 1.0."""
    image = OMEZarrImage(data=create_data((32, 256, 256)), axes="zyx")
    assert all(image.scale[d] == 1.0 for d in "zyx")


@pytest.mark.parametrize(
    "array_constructor", [np.array, da.from_array], ids=["numpy", "dask"]
)
def test_image_class_writer_array_constructor(tmp_path, array_constructor):
    """
    Check whether the writer accepts different array types.
    """
    shape = (2, 3, 4)
    axes = "cyx"
    scale = {"c": 1.0, "y": 0.5, "x": 0.5}
    data = array_constructor(create_data(shape))
    image = OMEZarrImage(data=data, axes=axes, scale=scale)
    ms = OMEZarrMultiscale.from_singlescale(
        image=image, scale_factors=None, method=None
    )

    grp_path = tmp_path / "test"
    ms.to_ome_zarr(group=str(grp_path), overwrite=True)

    out = zarr.open_group(grp_path)
    node_metadata = out.attrs.get("ome", out.attrs)
    path0 = node_metadata["multiscales"][0]["datasets"][0]["path"]
    written = da.from_zarr(grp_path / path0)

    # Written pixel data should match the input,
    # whether given as numpy or dask.
    assert np.allclose(data, written[...].compute())


@pytest.mark.parametrize("storage_options_list", [True, False])
def test_image_class_writer_storage_options(tmp_path, image_dims, storage_options_list):
    """
    Chunk shapes on disk follow storage_options, whether given as a dict or a list.
    """
    shape, axes, scale = image_dims
    data = create_data(shape)
    image = OMEZarrImage(data=data, axes=axes, scale=scale)
    scale_factors = [{d: 2 if d in ("x", "y") else 1 for d in axes}]
    ms = OMEZarrMultiscale.from_singlescale(image=image, scale_factors=scale_factors)

    chunks = [tuple(min(32, s) for s in shape), tuple(min(16, s) for s in shape)]
    storage_options = (
        [{"chunks": c} for c in chunks] if storage_options_list else {"chunks": chunks[0]}
    )
    grp_path = tmp_path / "test"
    ms.to_ome_zarr(group=str(grp_path), storage_options=storage_options, overwrite=True)

    out = zarr.open_group(grp_path)
    node_metadata = out.attrs.get("ome", out.attrs)
    paths = [d["path"] for d in node_metadata["multiscales"][0]["datasets"]]
    node_data = [da.from_zarr(grp_path / p) for p in paths]

    for level, nd_array in enumerate(node_data):
        expected = chunks[level] if storage_options_list else chunks[0]
        first_chunk = tuple(c[0] for c in nd_array.chunks)
        assert first_chunk == _retuple(expected, nd_array.shape)


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"], ids=["V04", "V05", "V06"])
def test_image_class_writer_labels(tmp_path, image_dims, version):
    """
    Labels written alongside an image can be read back.
    """
    shape, axes, scale = image_dims
    data = create_data(shape)
    data_labels = (data > data.mean()).astype(np.uint8)

    labels_name = "test_labels"
    image = OMEZarrImage(data=data, axes=axes, scale=scale)
    labels = OMEZarrImage(data=data_labels, axes=axes, scale=scale, name=labels_name)
    labels_multiscales = OMEZarrLabels(image=labels, scale_factors=None)
    ms = OMEZarrMultiscale.from_singlescale(
        image=image, scale_factors=None, method=None, labels=labels_multiscales
    )

    grp_path = (
        tmp_path / "v3" / "test" if version.startswith(("0.5", "0.6")) else tmp_path / "test"
    )
    ms.to_ome_zarr(group=str(grp_path), version=version, overwrite=True)

    # Check if the labels group exists and contains the expected label name.
    label_group_attrs = zarr.open(f"{grp_path}/labels", mode="r").attrs
    if version == "0.5" or version.startswith("0.6"):
        label_group_attrs = label_group_attrs["ome"]
    assert "labels" in label_group_attrs
    assert labels_name in label_group_attrs["labels"]

    # Read back the multiscale and check if the labels are correctly associated.
    ms_read = OMEZarrMultiscale.from_ome_zarr(str(grp_path))
    assert labels_name in list(ms_read.labels.keys())
    if "c" in axes:
        assert ms_read.omero is not None
    assert ms_read.labels[labels_name].image_label is not None


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"], ids=["V04", "V05", "V06"])
@pytest.mark.parametrize(
    "shape",
    [(256, 256), (32, 256, 256), (2, 32, 256, 256), (1, 2, 1, 256, 256)],
    ids=["2D", "3D", "4D", "5D"],
)
def test_image_class_writer_transformations(tmp_path, shape, version):
    """
    Check that written (and downsampled) coordinateTransformations match the expected scale.
    """

    TRANSFORMATIONS = [
        [{"scale": [1, 1, 0.5, 0.18, 0.18], "type": "scale"}],
        [{"scale": [1, 1, 0.5, 0.36, 0.36], "type": "scale"}],
        [{"scale": [1, 1, 0.5, 0.72, 0.72], "type": "scale"}],
        [{"scale": [1, 1, 0.5, 1.44, 1.44], "type": "scale"}],
        [{"scale": [1, 1, 0.5, 2.88, 2.88], "type": "scale"}],
    ]

    axes = "tczyx"[-len(shape) :]
    axes_scale = TRANSFORMATIONS[0][0]["scale"][-len(shape) :]
    scale_factors = [
        {str(d): 2**i if d in ("x", "y") else 1.0 for d in axes}
        for i in range(1, len(TRANSFORMATIONS))
    ]

    data = create_data(shape)
    image = OMEZarrImage(data=data, axes=axes, scale=dict(zip(axes, axes_scale)))
    ms = OMEZarrMultiscale.from_singlescale(image=image, scale_factors=scale_factors)

    grp_path = (
        tmp_path / "v3" / "test" if version.startswith(("0.5", "0.6")) else tmp_path / "test"
    )
    ms.to_ome_zarr(group=str(grp_path), version=version, overwrite=True)

    out = zarr.open_group(grp_path)
    node_metadata = out.attrs.get("ome", out.attrs)
    paths = [d["path"] for d in node_metadata["multiscales"][0]["datasets"]]
    node_data = [da.from_zarr(grp_path / path) for path in paths]

    for level, nd_array in enumerate(node_data):
        ds = node_metadata["multiscales"][0]["datasets"][level]
        if level == 0:
            # check first written scale values explicitly match those in TRANSFORMATIONS
            for d in axes:
                if version.startswith(("0.4", "0.5")):
                    tf = ds["coordinateTransformations"][0]
                    assert tf["scale"][axes.index(d)] == axes_scale[axes.index(d)]
                elif version.startswith("0.6"):
                    tf = ds["coordinateTransformations"][0]["transformations"][0]
                    assert tf["scale"][axes.index(d)] == axes_scale[axes.index(d)]
            continue

        # first calculate relative factors between this and previous level
        relative_factors = {
            d: node_data[0].shape[axes.index(d)] / nd_array.shape[axes.index(d)]
            for d in axes
        }

        # then convert into corresponding scale values
        expected_scale = {
            d: axes_scale[axes.index(d)] * relative_factors[d] for d in axes
        }

        # make sure we are doing this correctly for dimensions that
        # are not supposed to be downsampled
        if "t" in axes:
            assert expected_scale["t"] == 1.0
        if "c" in axes:
            assert expected_scale["c"] == 1.0
        if "z" in axes:
            assert relative_factors["z"] == 1.0

        # retrieve written scale factors from metadata and check they match expected
        if version.startswith(("0.4", "0.5")):
            cts = ds["coordinateTransformations"]
        elif version.startswith("0.6"):
            cts = ds["coordinateTransformations"][0]["transformations"]
        assert len(cts) == 2
        transf = cts[0]
        assert transf["type"] == "scale"
        for d in axes:
            assert transf["scale"][axes.index(d)] == expected_scale[d]