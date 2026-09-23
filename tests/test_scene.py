import logging
from collections.abc import Callable
from typing import Any, Self, TypedDict

import numpy as np
import ome_zarr_models.v06.coordinate_transforms as ozmt
import pytest
import transformnd as tnd
import zarr
from ome_zarr_models.v06.coordinate_transforms import (
    AnyTransform,
    CoordinateSystem,
    CoordinateSystemIdentifier,
)
from pydantic import TypeAdapter

from ome_zarr import OMEZarrImage, OMEZarrMultiscale, OMEZarrScene
from ome_zarr.classes.tnd_utils import (
    UnsupportedTransformation,
    _ozmp_tf_to_tnd,
    _setup_vectorfield_args,
)
from ome_zarr.utils import download

logger = logging.getLogger(__name__)
transform_adapter = TypeAdapter(AnyTransform)

TRANSFORMS = [
    {"type": "scale", "scale": (1.0, 1.0)},
    {"type": "translation", "translation": (0.0, 0.0)},
    {"type": "rotation", "rotation": ((0.0, -1.0), (1.0, 0.0))},
    {"type": "affine", "affine": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))},
    {"type": "mapAxis", "mapAxis": (1, 0)},
    {
        "type": "sequence",
        "transformations": [
            {"type": "scale", "scale": (1.0, 1.0)},
            {"type": "translation", "translation": (0.0, 0.0)},
        ],
    },
    {
        "type": "byDimension",
        "transformations": [
            {
                "inputAxes": [1],
                "outputAxes": [1],
                "transformation": {"type": "translation", "translation": (0.0,)},
            },
            {
                "inputAxes": [0],
                "outputAxes": [0],
                "transformation": {"type": "scale", "scale": (1.0,)},
            },
        ],
    },
]

TRANSFORMS = [transform_adapter.validate_python(t) for t in TRANSFORMS]


@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory with zarr v3 group for testing."""
    path = tmp_path / "data" / "v3"
    root_v3 = zarr.open_group(path, mode="w", zarr_format=3)
    root_v3.create_group("test")
    return path


def create_data(shape, dtype=np.uint8, mean_val=10):
    """Create dummy testing data of defined shape and type,
    with a given mean value."""
    rng = np.random.default_rng(0)
    return rng.poisson(mean_val, size=shape).astype(dtype)


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_create_scene_without_coordinate_systems(test_data_dir, transform):
    """
    Create a scene with two images and a single coordinate transformation
    between them.
    One of the images has an additional coordinate system besides the
    physical coordinate system.
    The data is saved and loaded back in the test.
    """
    shape = (64, 64)
    img_a = OMEZarrImage(
        data=create_data(shape),
        name="imageA",
        axes=["y", "x"],
        scale={"y": 1.0, "x": 1.0},
    )

    img_b = OMEZarrImage(
        data=create_data(shape),
        name="imageB",
        axes=["y", "x"],
        scale={"y": 1.0, "x": 1.0},
    )

    additional_cs = CoordinateSystem.model_validate(
        {
            "name": "additional",
            "axes": [
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
        }
    )

    additional_tf = TypeAdapter(AnyTransform).validate_python(
        {
            "type": "rotation",
            "rotation": ((0.0, -1.0), (1.0, 0.0)),
            "input": {"name": "physical"},
            "output": {"name": "additional"},
        }
    )

    img_a_ms = OMEZarrMultiscale(image=img_a)
    img_b_ms = OMEZarrMultiscale(
        image=img_b,
        coordinate_systems=[additional_cs],
        coordinate_transformations=[additional_tf],
    )

    # avoid leaking transform mutations into other tests
    transform = transform.copy()
    transform = transform.model_copy(
        update={
            "input": CoordinateSystemIdentifier(name="physical", path="imageA"),
            "output": CoordinateSystemIdentifier(name="physical", path="imageB"),
        }
    )

    scene = OMEZarrScene(
        images=[img_a_ms, img_b_ms],
        coordinate_transformations=[transform],
    )

    scene.to_ome_zarr(test_data_dir / "test_scene.zarr", overwrite=True)

    # test "downloading" the scene elsewhere
    download(
        test_data_dir / "test_scene.zarr", test_data_dir / "test_scene_downloaded.zarr"
    )

    # check that the graph is created correctly
    assert scene._graph is not None

    # check that the graph has the correct number of nodes
    # (aka coordinate systems)
    assert len(scene._graph.graph.nodes) == 3

    # traverse graph
    tf = scene._graph.get_sequence((img_a.name, "physical"), (img_b.name, "physical"))

    # make sure transform matrix is square
    # bydimension transforms cannot easily be expressed as matrix
    if transform.type != "byDimension":
        affine = tf.simplify().to_affine().matrix
        assert affine.shape[0] == affine.shape[1]

    # check that the transform graph can be traversed (i.e. transform is not None)
    assert tf is not None

    # write to disk and read back
    scene.to_ome_zarr(str(test_data_dir / "test_scene.zarr"), overwrite=True)
    scene_read = OMEZarrScene.from_ome_zarr(str(test_data_dir / "test_scene.zarr"))

    # check that the graph is created correctly on read
    assert scene_read._graph is not None
    assert len(scene_read._graph.graph.nodes) == 3

    # open the zarr group and check the metadata
    # and check that the correct metadata fields are present in the store
    zarr_group = zarr.open_group(str(test_data_dir / "test_scene.zarr"), mode="r")
    assert "ome" in zarr_group.attrs
    ome_metadata = zarr_group.attrs["ome"]
    assert "scene" in ome_metadata
    assert "version" in ome_metadata
    assert ome_metadata["version"] == "0.6"

    # check transforms
    assert "coordinateTransformations" in ome_metadata["scene"]
    assert len(ome_metadata["scene"]["coordinateTransformations"]) == 1

    transform_md = ome_metadata["scene"]["coordinateTransformations"][0]

    # make sure that the loaded transform is the same as the original
    assert transform_md == transform.model_dump(exclude_unset=True, mode="json")


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_create_scene_with_coordinate_systems(test_data_dir, transform):
    """
    Create a scene with two images and three coordinate transformations
    between them. The data is saved and loaded back in the test.
    """
    shape = (64, 64)
    img_a = OMEZarrImage(
        data=create_data(shape),
        name="imageA",
        axes=["y", "x"],
        scale={"y": 1.0, "x": 1.0},
    )

    img_b = OMEZarrImage(
        data=create_data(shape),
        name="imageB",
        axes=["y", "x"],
        scale={"y": 1.0, "x": 1.0},
    )

    img_a_ms = OMEZarrMultiscale(image=img_a)
    img_b_ms = OMEZarrMultiscale(image=img_b)

    world1_cs = CoordinateSystem.model_validate(
        {
            "name": "world",
            "axes": [
                ax.model_dump() for ax in img_a_ms.metadata.coordinateSystems[0].axes
            ],
        }
    )
    world2_cs = CoordinateSystem.model_validate(
        {
            "name": "world2",
            "axes": [
                ax.model_dump() for ax in img_b_ms.metadata.coordinateSystems[0].axes
            ],
        }
    )

    transform1 = transform.copy()
    transform1 = transform1.model_copy(
        update={
            "input": CoordinateSystemIdentifier(name="physical", path="imageA"),
            "output": CoordinateSystemIdentifier(name="world"),
        }
    )

    transform2 = transform.copy()
    transform2 = transform2.model_copy(
        update={
            "input": CoordinateSystemIdentifier(name="world"),
            "output": CoordinateSystemIdentifier(name="world2"),
        }
    )

    transform3 = transform.copy()
    transform3 = transform3.model_copy(
        update={
            "input": CoordinateSystemIdentifier(name="world2"),
            "output": CoordinateSystemIdentifier(name="physical", path="imageB"),
        }
    )

    scene = OMEZarrScene(
        images=[img_a_ms, img_b_ms],
        coordinate_transformations=[transform1, transform2, transform3],
        coordinate_systems=[world1_cs, world2_cs],
    )

    # check that the graph is created correctly
    # and has the correct number of nodes (coordinate systems)
    assert scene._graph is not None
    assert len(scene._graph.graph.nodes) == 4

    scene.to_ome_zarr(str(test_data_dir / "test_scene_with_cs.zarr"), overwrite=True)

    # test "downloading" the scene elsewhere
    download(
        test_data_dir / "test_scene_with_cs.zarr",
        test_data_dir / "test_scene_with_cs_downloaded.zarr",
    )

    # read in saved scene and check everything's there
    scene_read = OMEZarrScene.from_ome_zarr(
        str(test_data_dir / "test_scene_with_cs.zarr")
    )

    # check that the graph is created correctly on read
    # and has the correct number of nodes (coordinate systems)
    assert scene_read._graph is not None
    assert len(scene_read._graph.graph.nodes) == 4

    # open zarr group and check metadata
    zarr_group = zarr.open_group(
        str(test_data_dir / "test_scene_with_cs.zarr"), mode="r"
    )

    # make sure that the correct metadata fields are present in the store
    assert "ome" in zarr_group.attrs
    ome_metadata = zarr_group.attrs["ome"]
    assert "scene" in ome_metadata
    assert "version" in ome_metadata
    assert ome_metadata["version"] == "0.6"
    assert "coordinateSystems" in ome_metadata["scene"]
    assert len(ome_metadata["scene"]["coordinateSystems"]) == 2


def test_coordinate_system_retrieval(test_data_dir):
    """
    Create a scene with two images and three coordinate transformations
    between them. The data is saved and loaded back in the test.
    """
    shape = (64, 64)
    img_a = OMEZarrImage(
        data=create_data(shape),
        name="imageA",
        axes=["y", "x"],
        scale={"y": 1.0, "x": 1.0},
    )

    img_b = OMEZarrImage(
        data=create_data(shape),
        name="imageB",
        axes=["y", "x"],
        scale={"y": 1.0, "x": 1.0},
    )

    img_a_ms = OMEZarrMultiscale(image=img_a)
    img_b_ms = OMEZarrMultiscale(image=img_b)

    world1_cs = {
        "name": "world",
        "axes": [ax.model_dump() for ax in img_a_ms.metadata.coordinateSystems[0].axes],
    }
    world2_cs = {
        "name": "world2",
        "axes": [ax.model_dump() for ax in img_b_ms.metadata.coordinateSystems[0].axes],
    }

    transform1 = {"type": "scale", "scale": [1.0, 1.0]}
    transform1["input"] = {"name": "physical", "path": "imageA"}
    transform1["output"] = {"name": "world"}

    transform2 = {"type": "scale", "scale": [1.0, 1.0]}
    transform2["input"] = {"name": "world"}
    transform2["output"] = {"name": "world2"}

    transform3 = {"type": "scale", "scale": [1.0, 1.0]}
    transform3["input"] = {"name": "world2"}
    transform3["output"] = {"name": "physical", "path": "imageB"}

    scene = OMEZarrScene(
        images=[img_a_ms, img_b_ms],
        coordinate_transformations=[transform1, transform2, transform3],
        coordinate_systems=[world1_cs, world2_cs],
    )

    # check that the get_coordinate_system returns the correct
    # number and instances of coordinate systems
    all_cs = scene.get_coordinate_system()

    # Keys are (path, name) tuples
    assert ("", "world") in all_cs
    assert ("", "world2") in all_cs
    assert all_cs[("", "world")].name == "world"
    assert all_cs[("", "world2")].name == "world2"

    assert ("imageB", "physical") in all_cs
    assert all_cs[("imageB", "physical")].name == "physical"

    assert ("imageA", "physical") in all_cs
    assert all_cs[("imageA", "physical")].name == "physical"

    world_cs = scene.get_coordinate_system(path="")
    assert ("", "world") in world_cs
    assert ("", "world2") in world_cs
    assert ("imageA", "physical") not in world_cs
    assert ("imageB", "physical") not in world_cs

    imageA_cs = scene.get_coordinate_system(path="imageA")
    assert ("imageA", "physical") in imageA_cs
    assert ("", "world") not in imageA_cs
    assert ("imageB", "physical") not in imageA_cs

    all_physical_cs = scene.get_coordinate_system(name="physical")
    assert ("", "world") not in all_physical_cs
    assert ("imageA", "physical") in all_physical_cs
    assert ("imageB", "physical") in all_physical_cs


def test_appending_scene(test_data_dir):
    """
    Create a scene with two images and a single coordinate transformation
    between them. The data is saved and loaded back.
    We then append a third image to the scene and make sure the graph is updated.
    We then save the scene again and ensure that we are writing only what's new
    (new data and metadata)
    """
    img_a = OMEZarrImage(
        data=create_data((64, 64)),
        name="imageA",
        axes=["y", "x"],
        scale={"y": 1.0, "x": 1.0},
    )
    img_b = OMEZarrImage(
        data=create_data((64, 64)),
        name="imageB",
        axes=["y", "x"],
        scale={"y": 1.0, "x": 1.0},
    )
    img_c = OMEZarrImage(
        data=create_data((64, 64)),
        name="imageC",
        axes=["y", "x"],
        scale={"y": 1.0, "x": 1.0},
    )

    img_a_ms = OMEZarrMultiscale(image=img_a)
    img_b_ms = OMEZarrMultiscale(image=img_b)
    img_c_ms = OMEZarrMultiscale(image=img_c)

    world_cs = {
        "name": "world",
        "axes": [ax.model_dump() for ax in img_a_ms.metadata.coordinateSystems[0].axes],
    }

    tf1 = {
        "type": "scale",
        "scale": [1.0, 1.0],
        "input": {"name": "physical", "path": "imageA"},
        "output": {"name": "world"},
    }

    tf2 = {
        "type": "scale",
        "scale": [1.0, 1.0],
        "input": {"name": "world"},
        "output": {"name": "physical", "path": "imageB"},
    }

    tf3 = {
        "type": "scale",
        "scale": [1.0, 1.0],
        "input": {"name": "world"},
        "output": {"name": "physical", "path": "imageC"},
    }

    scene = OMEZarrScene(
        images=[img_a_ms, img_b_ms],
        coordinate_transformations=[tf1, tf2],
        coordinate_systems=[world_cs],
    )

    scene.to_ome_zarr(str(test_data_dir / "test_scene_append.zarr"), overwrite=True)

    # now we load the scene and append a new image to it
    scene_read = OMEZarrScene.from_ome_zarr(
        str(test_data_dir / "test_scene_append.zarr")
    )

    new_scene = OMEZarrScene(
        images=list(scene_read.images.values()) + [img_c_ms],
        coordinate_transformations=list(scene_read.coordinate_transformations) + [tf3],
        coordinate_systems=[world_cs],
    )

    new_scene.to_ome_zarr(
        str(test_data_dir / "test_scene_append.zarr"), overwrite=False
    )

    # check that the graph is built correctly
    assert new_scene._graph is not None
    assert len(new_scene._graph.graph.nodes) == 4

    # check that the data is written and not empty
    zarr_group = zarr.open_group(
        str(test_data_dir / "test_scene_append.zarr"), mode="r"
    )
    assert "imageA" in zarr_group and "s0" in zarr_group["imageA"]
    assert "imageB" in zarr_group and "s0" in zarr_group["imageB"]
    assert "imageC" in zarr_group and "s0" in zarr_group["imageC"]

    # check that the metadata is written and correct
    assert "ome" in zarr_group.attrs
    ome_metadata = zarr_group.attrs["ome"]
    assert "scene" in ome_metadata
    scene_metadata = ome_metadata["scene"]
    assert "coordinateTransformations" in scene_metadata
    assert "coordinateSystems" in scene_metadata
    assert len(scene_metadata["coordinateTransformations"]) == 3
    assert len(scene_metadata["coordinateSystems"]) == 1


def test_scene_with_displacements(test_data_dir):
    """
    Create a scene with two images and a single coordinate transformation
    between them. The data is saved and loaded back in the test.
    """
    shape = (64, 64)
    img_a = OMEZarrImage(
        data=create_data(shape),
        name="imageA",
        axes=["y", "x"],
        scale={"y": 1.0, "x": 1.0},
    )

    img_b = OMEZarrImage(
        data=create_data(shape),
        name="imageB",
        axes=["y", "x"],
        scale={"y": 1.0, "x": 1.0},
    )

    vector_field = np.zeros((2, 64, 64), dtype=np.float32)
    dfield_img = OMEZarrImage(
        data=vector_field,
        name="displacementField",
        axes=["c", "y", "x"],
        scale={"y": 1.0, "x": 1.0},
        axes_types={"c": "displacement", "y": "space", "x": "space"},
    )

    img_a_ms = OMEZarrMultiscale(image=img_a)
    img_b_ms = OMEZarrMultiscale(image=img_b)
    dfield_img_ms = OMEZarrMultiscale(image=dfield_img)

    transform = {
        "type": "displacements",
        "input": {"name": "physical", "path": "imageA"},
        "output": {"name": "physical", "path": "imageB"},
        "path": "coordinateTransformations/displacementField",
    }

    scene = OMEZarrScene(
        images=[img_a_ms, img_b_ms],
        coordinate_transformations=[transform],
        coordinate_displacements={"displacementField": dfield_img_ms},
    )

    save_grp = str(test_data_dir / "test_scene_displacement.zarr")
    scene.to_ome_zarr(save_grp, overwrite=True)

    # check that all subgroups are there
    group = zarr.open_group(save_grp, mode="r")
    assert "coordinateTransformations" in group
    assert "displacementField" in group["coordinateTransformations"]
    assert "imageA" in group
    assert "imageB" in group

    # check that the metadata of the displacement field is correct
    dfield_attrs = group["coordinateTransformations"]["displacementField"].attrs
    assert "ome" in dfield_attrs
    assert dfield_attrs["ome"]["version"] == "0.6"
    axes_md = dfield_attrs["ome"]["multiscales"][0]["coordinateSystems"][0]["axes"]
    assert axes_md[0]["type"] == "displacement"
    assert axes_md[0]["discrete"]

    # read displacements back in and check that the (meta)data is correct
    scene_read = OMEZarrScene.from_ome_zarr(save_grp)
    assert "displacementField" in scene_read.coordinate_displacements

    dfield_img = scene_read.coordinate_displacements["displacementField"]
    for image in dfield_img.images:
        assert image.axes_types["c"] == "displacement"
    assert dfield_img.metadata.coordinateSystems[0].axes[0].discrete


class CaseBuilder:
    def __init__(
        self, params: list[str], id_fn: Callable[[tuple], str] | None = None
    ) -> None:
        self.params = params
        self.cases: list[tuple] = []
        self.id_fn = id_fn
        self.ids: list[str] = []

    def get_id(self, args: tuple, override: str | None = None) -> str:
        if override:
            return override
        if self.id_fn is None:
            return f"case{len(self.cases)}"
        return self.id_fn(args)

    def add(self, *args, id: str | None = None) -> Self:  # noqa: A002
        assert len(self.params) == len(args)
        self.ids.append(self.get_id(args, id))
        self.cases.append(args)
        return self

    def parametrize(self, test_fn: Callable):
        return pytest.mark.parametrize(self.params, self.cases, ids=self.ids)(test_fn)


class OzmpTfKwargs(TypedDict):
    zarr_context: str
    source_ndim: int | None
    target_ndim: int | None
    coordinate_displacements: dict[str, Any]


@ (
    CaseBuilder(["ozm_transform", "kwargs"])
    .add(ozmt.Identity(), {"source_ndim": 3}, id="identity")
    .add(ozmt.Scale(scale=(2, 3)), {}, id="scale")
    .add(ozmt.Translation(translation=(2, 3)), {}, id="translation")
    .add(ozmt.Affine(affine=((1, 0, 0), (0, 1, 0))), {}, id="affine")
    .add(ozmt.Affine(affine=((1, 0, 0), (0, 1, 0), (0, 0, 1))), {}, id="affine_square")
    .add(ozmt.MapAxis(mapAxis=(2, 1, 0)), {}, id="mapAxis")
    .add(
        ozmt.ProjectAxis(createdOutputs=(1,), droppedInputs=(0,)),
        {"source_ndim": 3},
        id="projectAxis",
    )
    .add(ozmt.Rotation(rotation=((1, 0), (0, 1))), {}, id="rotation")
    .add(
        ozmt.ByDimension(
            transformations=(
                ozmt.ByDimensionTransform(
                    inputAxes=(0, 2), outputAxes=(0, 2), transformation=ozmt.Identity()
                ),
                ozmt.ByDimensionTransform(
                    inputAxes=(1,),
                    outputAxes=(1,),
                    transformation=ozmt.Translation(translation=(10,)),
                ),
            )
        ),
        {},
        id="byDimension",
    )
    .add(
        ozmt.Sequence(
            transformations=(
                ozmt.Translation(translation=(2, 3)),
                ozmt.Scale(scale=(10, 100)),
            )
        ),
        {},
        id="sequence",
    )
).parametrize
def test_convert_transformations(
    ozm_transform: ozmt.AnyTransform, kwargs: OzmpTfKwargs
):
    defaults: OzmpTfKwargs = {
        "zarr_context": "",
        "source_ndim": None,
        "target_ndim": None,
        "coordinate_displacements": dict(),
    }
    defaults.update(kwargs)
    t = _ozmp_tf_to_tnd(ozm_transform, **defaults)
    assert t is not None
    check_transforms_equivalent(ozm_transform, t)


def check_transforms_equivalent(
    ozm_transform: ozmt.AnyTransform, tnd_transform: tnd.Transform
):
    ndim = tnd_transform.ndims.source
    rng = np.random.default_rng(1991)
    in_coords = rng.uniform(-100, 100, (10, ndim))
    tnd_results = tnd_transform.apply(in_coords)
    try:
        ozm_results = np.array(
            [ozm_transform.transform_point(row) for row in in_coords]
        )
        assert tnd_results == pytest.approx(ozm_results)
    except NotImplementedError:
        logger.info(
            "ome-zarr-models does not implement coordinate transformations for %s",
            ozm_transform,
        )
    except ValueError as e:
        if isinstance(ozm_transform, ozmt.Affine) and "dimensionality" in str(e):
            pytest.xfail(
                "Affine dimensionality check bug resolved in https://github.com/ome-zarr-models/ome-zarr-models-py/pull/481"
            )
        raise

    tnd_inv = tnd_transform.invert()
    try:
        if ozm_transform.has_inverse:
            assert (
                tnd_inv is not None
            ), "ome-zarr-models implements inversion but transformnd does not"
    except NotImplementedError:
        pass


def test_setup_vectorfield():
    t = ozmt.Displacements(
        input=CoordinateSystemIdentifier(name="in_name", path="in_path"),
        output=CoordinateSystemIdentifier(name="out_name", path="out_path"),
        path="dpath",
    )

    with pytest.raises(UnsupportedTransformation):
        _setup_vectorfield_args(t, "", dict())


@pytest.mark.parametrize(
    ("transform",),
    [
        (ozmt.Identity(),),
        (ozmt.Sequence(transformations=tuple()),),
        (ozmt.ProjectAxis(droppedInputs=(0,)),),
    ],
)
def test_no_dimensionality(transform: AnyTransform):
    with pytest.raises(UnsupportedTransformation):
        _ozmp_tf_to_tnd(transform, "", None, None, dict())


@pytest.mark.parametrize(("seq_len",), [(1,), (2,), (3,), (4,)])
def test_convert_sequence(seq_len):
    scale = ozmt.Scale(scale=(2, 3))
    _ozmp_tf_to_tnd(
        ozmt.Sequence(transformations=(scale,) * seq_len), ",", None, None, dict()
    )


if __name__ == "__main__":
    pytest.main([__file__])
