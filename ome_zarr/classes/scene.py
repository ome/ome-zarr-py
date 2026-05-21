# the class for storage representation, not exposed to the user
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import zarr
from ome_zarr_models._v06.coordinate_transforms import (
    AnyTransform,
    CoordinateSystem,
)
from ome_zarr_models._v06.scene import SceneAttrs
from pydantic import TypeAdapter
from zarr.storage import StoreLike

from .image import NgffMultiscales


# the class exposed to the user
@dataclass(kw_only=True)
class NgffScene:
    images: list[NgffMultiscales]
    metadata: SceneAttrs = field(init=False, default=None)
    coordinate_transformations: Sequence[AnyTransform] | list[dict[str, Any]]
    coordinate_systems: Sequence[CoordinateSystem] | Sequence[dict[str, Any]] | None = (
        None
    )
    _written_image_names: set[str] = field(default_factory=set, init=False)

    def __post_init__(self):

        # parse coordinate systems and transforms
        self.coordinate_systems = self._parse_coordinate_systems(
            self.coordinate_systems
        )
        self.coordinate_transformations = self._parse_transforms(
            self.coordinate_transformations
        )

        self.metadata = SceneAttrs(
            coordinateSystems=self.coordinate_systems,
            coordinateTransformations=self.coordinate_transformations,
        )

        # self._graph = nx.DiGraph()

        # if self.coordinate_systems is not None:
        #     for cs in self.coordinate_systems:
        #         if hasattr(cs, "path") and cs.path is None:
        #             self._graph.add_node((None, cs.name))

        # for img in self.images:
        #     # add all coordinate systems as nodes
        #     for cs in img.metadata.coordinateSystems:
        #         node_id = (img.metadata.name, cs.name)
        #         self._graph.add_node(node_id)

        #     for ds in img.metadata.datasets:
        #         # add all datasets as nodes
        #         node_id = (img.metadata.name, ds.path)
        #         self._graph.add_node(node_id)

        #         # add scale transformations from dataset as edges
        #         transform = ds.coordinateTransformations
        #         self._graph.add_edge(
        #             (img.metadata.name, ds.path),
        #             (img.metadata.name, ds.coordinateTransformations[0].output),
        #             transform=transform,
        #         )

        #     # add additional transformations from image metadata as edges
        #     if img.metadata.coordinateTransformations:
        #         for tf in img.metadata.coordinateTransformations:
        #             self._graph.add_edge(
        #                 (img.metadata.name, tf.input),
        #                 (img.metadata.name, tf.output),
        #                 transform=tf,
        #             )

        # # add scene-level transformations as edges between coordinate systems of different images
        # for tf in self.transformations:
        #     self._graph.add_edge(
        #         (tf.input.path, tf.input.name),
        #         (tf.output.path, tf.output.name),
        #         transform=tf)

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

        # Create a subgroup for each image using its name
        for img in tqdm.tqdm(self.images, desc="Writing images"):
            img_name = str(img.metadata.name)

            # Skip if already written (incremental mode)
            if not overwrite and img_name in self._written_image_names:
                continue

            # Write the image
            subgroup = zarr_group.create_group(img_name, overwrite=not overwrite)
            img.to_ome_zarr(subgroup)
            self._written_image_names.add(img_name)

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

        zarr_group = zarr.open(store, mode="r")

        # Load all image subgroups
        images = []
        for img_name in zarr_group.group_keys():
            img_group = zarr_group[img_name]
            # Assume images have their own to_ome_zarr-like interface
            # You may need to adapt based on your NgffMultiscales.from_ome_zarr implementation
            img = NgffMultiscales.from_ome_zarr(img_group)
            images.append(img)

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
        scene = object.__new__(cls)

        # Set fields directly using object.__setattr__ to bypass custom __setattr__
        object.__setattr__(scene, "images", images)
        object.__setattr__(scene, "coordinate_transformations", tuple(transformations))
        object.__setattr__(
            scene,
            "coordinate_systems",
            tuple(coordinate_systems) if coordinate_systems else None,
        )
        object.__setattr__(
            scene, "_written_image_names", {img.metadata.name for img in images}
        )

        # Now set metadata
        object.__setattr__(
            scene,
            "metadata",
            SceneAttrs(
                coordinateSystems=scene.coordinate_systems,
                coordinateTransformations=scene.coordinate_transformations,
            ),
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
