# the class for storage representation, not exposed to the user
import os
from collections.abc import Sequence
from typing import Any

import transformnd as tnd
import zarr
from ome_zarr_models.v06.coordinate_transforms import (
    AnyTransform,
    CoordinateSystem,
)
from ome_zarr_models.v06.scene import SceneAttrs
from pydantic import TypeAdapter
from zarr.storage import StoreLike

from .image import OMEZarrMultiscale


class OMEZarrScene:
    def __init__(
        self,
        images: list[OMEZarrMultiscale] | dict[str, OMEZarrMultiscale],
        coordinate_transformations: Sequence[AnyTransform] | list[dict[str, Any]],
        coordinate_systems: (
            Sequence[CoordinateSystem] | Sequence[dict[str, Any]] | None
        ) = None,
    ):
        """
        Parameters
        ----------
        images : list[OMEZarrMultiscale] | dict[str, OMEZarrMultiscale]
            Either a list of images (keyed internally by metadata.name) or a dict
            mapping zarr group paths to images. The dict form gives explicit control
            over the paths where images will be stored in the zarr hierarchy.
        """
        # Coerce list to dict keyed by metadata.name
        if isinstance(images, list):
            self.images = {str(img.metadata.name): img for img in images}
        else:
            self.images = images

        # parse coordinate systems and transforms
        self.coordinate_systems = self._parse_coordinate_systems(coordinate_systems)
        self.coordinate_transformations = self._parse_transforms(
            coordinate_transformations
        )

        self.metadata = SceneAttrs(
            coordinateSystems=self.coordinate_systems,
            coordinateTransformations=self.coordinate_transformations,
        )

        self._build_graph()
        self._written_image_names = set()

    def _build_graph(self):
        self._graph = tnd.graph.TransformGraph()
        # Add scene-level transformations (empty context = root level)
        for tf in self.coordinate_transformations:
            if tf.type == "sequence":
                tnd_transform = _ozmp_tf_to_tnd(tf).simplify()
            else:
                tnd_transform = [_ozmp_tf_to_tnd(tf)]
            self._graph.add_transforms(tnd_transform)

            # check if input/output are defined
            subgroups = []
            if tf.input.path is not None:
                subgroups.append(tf.input.path)
            if tf.output.path is not None:
                subgroups.append(tf.output.path)

            for subgroup in subgroups:
                img = self.images.get(subgroup)
                if img is None:
                    # Image not found in scene - skip or warn
                    continue
                if img.metadata.coordinateTransformations:
                    for img_tf in img.metadata.coordinateTransformations:
                        ind_transform = _ozmp_tf_to_tnd(img_tf, zarr_context=subgroup)
                        self._graph.add_transforms([ind_transform])

    def to_ome_zarr(self, store: StoreLike, overwrite: bool = False):
        """
        Write scene to OME-Zarr format.

        Parameters
        ----------
        store: StoreLike
            A zarr-compatible storage backend (e.g., directory path, in-memory store, etc.)
        overwrite: bool
            If True, overwrite all images in the store with the current state of the scene.
            If False, only write new images that haven't been written before. Existing images in the store will be left unchanged.

        """
        import shutil

        import tqdm

        from ..utils import _recursive_pop_nones

        if overwrite and os.path.exists(str(store)):
            # Clear the store if it already exists and we're not doing incremental writes
            shutil.rmtree(str(store))

        # Open or create zarr group
        mode = "a" if overwrite else "w"
        zarr_group = zarr.open(store, mode=mode)

        # Create a subgroup for each image using its path key
        for img_path, img in tqdm.tqdm(self.images.items(), desc="Writing images"):
            # Skip if already written (incremental mode)
            if not overwrite and img_path in self._written_image_names:
                continue

            # Write the image
            subgroup = zarr_group.create_group(img_path, overwrite=not overwrite)
            img.to_ome_zarr(subgroup, overwrite=overwrite, version="0.6.dev4")
            self._written_image_names.add(img_path)

        # Always update scene metadata
        metadata_dict = self.metadata.model_dump()
        metadata_dict = _recursive_pop_nones(metadata_dict)

        zarr_group.attrs["ome"] = {"scene": metadata_dict, "version": "0.6"}

    @classmethod
    def from_ome_zarr(cls, store: StoreLike):
        """
        Load an existing scene from OME-Zarr format.

        Args:
            path: Path to the OME-Zarr scene

        Returns:
            NgffScene instance with images and metadata loaded from disk
        """
        tf_adapter = TypeAdapter(AnyTransform)

        # Handle both StoreLike (string, dict, etc.) and zarr.Group objects
        if isinstance(store, zarr.Group):
            zarr_group = store
        else:
            zarr_group = zarr.open(store, mode="r")

        # Load all image subgroups, keyed by their zarr path
        images = {}
        for img_path in zarr_group.group_keys():
            img_group = zarr_group[img_path]
            img = OMEZarrMultiscale.from_ome_zarr(img_group)
            images[img_path] = img

        # Load scene metadata
        ome_metadata = zarr_group.attrs.get("ome", {})
        scene_metadata = ome_metadata.get("scene", {})

        # Note: Reconstructing Transform and CoordinateSystemIdentifier objects from dicts
        # may require additional deserialization logic depending on your models

        transformations = []
        for tf in scene_metadata.get("coordinateTransformations", []):
            transformations.append(tf_adapter.validate_python(tf))

        if "coordinateSystems" in scene_metadata:
            coordinate_systems = [
                CoordinateSystem.model_validate(cs)
                for cs in scene_metadata.get("coordinateSystems", [])
                if cs
            ]
        else:
            coordinate_systems = None

        # Use object.__new__ to create instance without triggering __init__ and __setattr__
        scene = OMEZarrScene(
            images=images,
            coordinate_transformations=transformations,
            coordinate_systems=coordinate_systems,
        )

        return scene

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "coordinate_transformations":
            # Update metadata when coordinate transformations are set
            parsed_transforms = self._parse_transforms(value)
            super().__setattr__(name, parsed_transforms)
            # Only update metadata if it exists (not during initial construction)
            if hasattr(self, "metadata") and self.metadata is not None:
                self.metadata = self.metadata.model_copy(
                    update={"coordinateTransformations": parsed_transforms}
                )

        elif name == "coordinate_systems":
            # Update metadata when coordinate systems are set
            parsed_coordinate_systems = self._parse_coordinate_systems(value)
            super().__setattr__(name, parsed_coordinate_systems)
            # Only update metadata if it exists (not during initial construction)
            if hasattr(self, "metadata") and self.metadata is not None:
                self.metadata = self.metadata.model_copy(
                    update={"coordinateSystems": parsed_coordinate_systems}
                )

        else:
            # Default behavior for all other attributes
            super().__setattr__(name, value)

    @staticmethod
    def _parse_transforms(
        transforms: Sequence[AnyTransform] | list[dict[str, Any]],
    ) -> tuple[AnyTransform, ...]:
        """
        Helper method to parse a sequence of coordinate transformations that may be provided as either
        AnyTransform instances or dictionaries.
        This ensures that all transformations are stored as AnyTransform objects in the scene metadata.
        """
        tf_adapter = TypeAdapter(AnyTransform)
        parsed_transforms = []

        for tf in transforms:
            if isinstance(tf, dict):
                parsed_transforms.append(tf_adapter.validate_python(tf))
            else:
                parsed_transforms.append(tf)

        return tuple(parsed_transforms)

    @staticmethod
    def _parse_coordinate_systems(
        coordinate_systems: (
            Sequence[CoordinateSystem] | Sequence[dict[str, Any]] | None
        ),
    ) -> tuple[CoordinateSystem, ...] | None:
        """
        Helper method to parse a sequence of coordinate systems that may be provided as either
        CoordinateSystem instances or dictionaries.
        This ensures that all coordinate systems are stored as CoordinateSystem objects in the scene metadata.
        If coordinate_systems is None, it will be returned as None.
        """
        if coordinate_systems is None:
            return None

        parsed_coordinate_systems = []
        for cs in coordinate_systems:
            if isinstance(cs, dict):
                parsed_coordinate_systems.append(CoordinateSystem.model_validate(cs))
            elif isinstance(cs, CoordinateSystem):
                parsed_coordinate_systems.append(cs)

        return tuple(parsed_coordinate_systems)


def _ozmp_tf_to_tnd(
    transform: AnyTransform, zarr_context: str = ""
) -> tnd.base.Transform:
    """
    Convert an OME-Zarr coordinate transformation to a transformnd Transform object.
    This is a placeholder function and will need to be implemented based on the specific types of transformations you expect to encounter in OME-Zarr metadata.
    """
    tnd_transform = None
    # Example for an affine transformation (this will depend on the actual structure of AnyTransform)
    if transform.type == "affine":
        tnd_transform = tnd.transforms.Affine.from_linear_map(transform.affine)
    elif transform.type == "scale":
        tnd_transform = tnd.transforms.Scale(transform.scale)
    elif transform.type == "translation":
        tnd_transform = tnd.transforms.Translate(transform.translation)
    elif transform.type == "rotation":
        tnd_transform = tnd.transforms.Affine.from_linear_map(transform.rotation)
    elif transform.type == "sequence":
        sub_transformations = transform.transformations
        tnd_sub_transforms = [
            _ozmp_tf_to_tnd(sub_tf, zarr_context) for sub_tf in sub_transformations
        ]
        tnd_transform = tnd.base.TransformSequence(tnd_sub_transforms)

    if transform.input is not None and tnd_transform is not None:
        input_path = transform.input.path if transform.input.path is not None else ""
        output_path = transform.output.path if transform.output.path is not None else ""

        # zarr_context prepends path with relative path from root
        # to keep track of global location of coordinate systems in the zarr store
        if zarr_context != "" and input_path != "":
            input_path = f"{zarr_context}/{input_path}"
        elif zarr_context != "":
            input_path = zarr_context

        if zarr_context != "" and output_path != "":
            output_path = f"{zarr_context}/{output_path}"
        elif zarr_context != "":
            output_path = zarr_context

        tnd_transform.spaces = tnd.Spaces(
            f"{input_path}:{transform.input.name}",
            f"{output_path}:{transform.output.name}",
        )

    return tnd_transform
