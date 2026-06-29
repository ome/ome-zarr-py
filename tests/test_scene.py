import numpy as np
from ome_zarr import OMEZarrImage, OMEZarrMultiscale, OMEZarrScene
import pytest
import pathlib
import zarr
import transformnd as tnd

class TestScene:
    TRANSFORMS = [
        {"type": "scale", "scale": [1.0, 1.0]},
        {"type": "translation", "translation": [0.0, 0.0]},
        {"type": "rotation", "rotation": [[0.0, -1.0], [1.0, 0.0]]},
        {"type": "affine", "affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]},
        {"type": "mapAxis", "mapAxis": [1, 0]}
    ]
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = pathlib.Path(tmpdir.mkdir("data"))

        # let's create zarr v3 group too...
        self.path = self.path / "v3"
        root_v3 = zarr.open_group(self.path, mode="w", zarr_format=3)
        self.group_v3 = root_v3.create_group("test")

    def create_data(self, shape, dtype=np.uint8, mean_val=10):
        rng = np.random.default_rng(0)
        return rng.poisson(mean_val, size=shape).astype(dtype)

    @pytest.mark.parametrize("transform", TRANSFORMS)
    def test_create_scene_without_coordinate_systems(self, transform):
        shape = (64, 64)
        img_a = OMEZarrImage(
            data=self.create_data(shape),
            name="imageA",
            axes=["y", "x"],
            scale={"y": 1.0, "x": 1.0},
        )

        img_b = OMEZarrImage(
            data=self.create_data(shape),
            name="imageB",
            axes=["y", "x"],
            scale={"y": 1.0, "x": 1.0},
        )

        img_a_ms = OMEZarrMultiscale(image=img_a)
        img_b_ms = OMEZarrMultiscale(image=img_b)

        transform["input"] = {"name": "physical", "path": "imageA"}
        transform["output"] = {"name": "physical", "path": "imageB"}

        scene = OMEZarrScene(
            images=[img_a_ms, img_b_ms],
            coordinate_transformations=[transform],
        )

        scene.to_ome_zarr("test_scene.zarr", overwrite=True)

        assert scene._graph is not None
        assert len(scene._graph.graph.nodes) == 2

        # traverse graph
        tf = scene._graph.get_sequence(f"{img_a.name}:physical", f"{img_b.name}:physical")
        assert tf is not None

        # write to disk and read back
        scene.to_ome_zarr(str(self.path / "test_scene.zarr"), overwrite=True)
        scene_read = OMEZarrScene.from_ome_zarr(str(self.path / "test_scene.zarr"))

        assert scene_read._graph is not None
        assert len(scene_read._graph.graph.nodes) == 2

        # open the zarr group and check the metadata
        zarr_group = zarr.open_group(str(self.path / "test_scene.zarr"), mode="r")
        assert "ome" in zarr_group.attrs
        ome_metadata = zarr_group.attrs["ome"]
        assert "scene" in ome_metadata
        assert "version" in ome_metadata and ome_metadata["version"] == "0.6.dev4"

        # check transforms
        assert "coordinateTransformations" in ome_metadata["scene"]
        assert len(ome_metadata["scene"]["coordinateTransformations"]) == 1

        transform_md = ome_metadata["scene"]["coordinateTransformations"][0]
        assert transform_md == transform

    @pytest.mark.parametrize("transform", TRANSFORMS)
    def test_create_scene_with_coordinate_systems(self, transform):
        shape = (64, 64)
        img_a = OMEZarrImage(
            data=self.create_data(shape),
            name="imageA",
            axes=["y", "x"],
            scale={"y": 1.0, "x": 1.0},
        )

        img_b = OMEZarrImage(
            data=self.create_data(shape),
            name="imageB",
            axes=["y", "x"],
            scale={"y": 1.0, "x": 1.0},
        )

        img_a_ms = OMEZarrMultiscale(image=img_a)
        img_b_ms = OMEZarrMultiscale(image=img_b)

        world1_cs = {"name": "world", "axes": [ax.model_dump() for ax in img_a_ms.metadata.coordinateSystems[0].axes]}
        world2_cs = {"name": "world2", "axes": [ax.model_dump() for ax in img_b_ms.metadata.coordinateSystems[0].axes]}

        transform1 = transform.copy()
        transform1["input"] = {"name": "physical", "path": "imageA"}
        transform1["output"] = {"name": "world"}

        transform2 = transform.copy()
        transform2["input"] = {"name": "world"}
        transform2["output"] = {"name": "world2"}

        transform3 = transform.copy()
        transform3["input"] = {"name": "world2"}
        transform3["output"] = {"name": "physical", "path": "imageB"}

        scene = OMEZarrScene(
            images=[img_a_ms, img_b_ms],
            coordinate_transformations=[transform1, transform2, transform3],
            coordinate_systems=[world1_cs, world2_cs]
        )

        assert scene._graph is not None
        assert len(scene._graph.graph.nodes) == 4

        scene.to_ome_zarr(str(self.path / "test_scene_with_cs.zarr"), overwrite=True)
        scene_read = OMEZarrScene.from_ome_zarr(str(self.path / "test_scene_with_cs.zarr"))

        assert scene_read._graph is not None
        assert len(scene_read._graph.graph.nodes) == 4

        # open zarr group and check metadata
        zarr_group = zarr.open_group(str(self.path / "test_scene_with_cs.zarr"), mode="r")
        assert "ome" in zarr_group.attrs
        ome_metadata = zarr_group.attrs["ome"]
        assert "scene" in ome_metadata
        assert "version" in ome_metadata and ome_metadata["version"] == "0.6.dev4"
        assert "coordinateSystems" in ome_metadata["scene"]
        assert len(ome_metadata["scene"]["coordinateSystems"]) == 2

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])