# the class for storage representation, not exposed to the user
import os
from dataclasses import dataclass, field

import networkx as nx
import zarr
from ome_zarr_models._v06.coordinate_transforms import (
    CoordinateSystemIdentifier,
    Transform,
    Translation,
)
from zarr.storage import StoreLike

from .image import NgffMultiscales


@dataclass
class SceneMetadata:
    coordinateTransformations: list[Transform]
    coordinateSystems: (
        tuple[CoordinateSystemIdentifier] | list[CoordinateSystemIdentifier]
    ) = field(default_factory=list)


# the class exposed to the user
@dataclass(kw_only=True)
class NgffScene:
    images: list[NgffMultiscales]
    transformations: list[Transform]
    coordinate_systems: (
        tuple[CoordinateSystemIdentifier] | list[CoordinateSystemIdentifier]
    ) = field(default_factory=list)
    _written_image_names: set[str] = field(default_factory=set, init=False)

    def __post_init__(self):

        self.metadata = SceneMetadata(
            coordinateTransformations=self.transformations,
            coordinateSystems=self.coordinate_systems,
        )

        self._graph = nx.DiGraph()

        for cs in self.coordinate_systems:
            if hasattr(cs, "path") and cs.path is None:
                self._graph.add_node((None, cs.name))

        for img in self.images:
            # add all coordinate systems as nodes
            for cs in img.metadata.coordinateSystems:
                node_id = (img.metadata.name, cs.name)
                self._graph.add_node(node_id)

            for ds in img.metadata.datasets:
                # add all datasets as nodes
                node_id = (img.metadata.name, ds.path)
                self._graph.add_node(node_id)

                # add scale transformations from dataset as edges
                transform = ds.coordinateTransformations
                self._graph.add_edge(
                    (img.metadata.name, ds.path),
                    (img.metadata.name, ds.coordinateTransformations[0].output),
                    transform=transform,
                )

            # add additional transformations from image metadata as edges
            if img.metadata.coordinateTransformations:
                for tf in img.metadata.coordinateTransformations:
                    self._graph.add_edge(
                        (img.metadata.name, tf.input),
                        (img.metadata.name, tf.output),
                        transform=tf,
                    )

        # add scene-level transformations as edges between coordinate systems of different images
        for tf in self.transformations:
            self._graph.add_edge(
                (tf.input.path, tf.input.name),
                (tf.output.path, tf.output.name),
                transform=tf,
            )

    def to_ome_zarr(self, store: StoreLike, incremental: bool = False):
        """
        Write scene to OME-Zarr format.

        Parameters
        ----------
        store: StoreLike
            A zarr-compatible storage backend (e.g., directory path, in-memory store, etc.)
        incremental: bool
            If True, only write new images that haven't been written before. Existing images in the store will be left unchanged.
            If False, overwrite all images in the store with the current state of the scene.

        """
        import shutil

        from .image import _recursive_pop_nones

        if not incremental and os.path.exists(str(store)):
            # Clear the store if it already exists and we're not doing incremental writes
            shutil.rmtree(str(store))

        # Open or create zarr group
        mode = "a" if incremental else "w"
        zarr_group = zarr.open(store, mode=mode)

        # Create a subgroup for each image using its name
        for img in self.images:
            img_name = str(img.metadata.name)

            # Skip if already written (incremental mode)
            if incremental and img_name in self._written_image_names:
                continue

            # Write the image
            subgroup = zarr_group.create_group(img_name, overwrite=not incremental)
            img.to_ome_zarr(subgroup)
            self._written_image_names.add(img_name)

        # Always update scene metadata
        metadata_dict = {
            "coordinateTransformations": [
                t.model_dump() for t in self.metadata.coordinateTransformations
            ],
            "coordinateSystems": (
                [cs.model_dump() for cs in self.metadata.coordinateSystems]
                if self.metadata.coordinateSystems
                else None
            ),
        }
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
            if tf["type"] == "translation":
                transformations.append(Translation.model_validate(tf))
        coordinate_systems = [
            CoordinateSystemIdentifier.model_validate(cs)
            for cs in scene_metadata.get("coordinateSystems", [])
            if cs
        ]

        scene = cls(
            images=images,
            transformations=transformations,
            coordinate_systems=coordinate_systems,
        )
        # Mark all existing images as already written
        scene._written_image_names = {img.metadata.name for img in images}

        return scene

    def add_transform(self, transform: Transform):
        self.transformations.append(transform)
        self.metadata = SceneMetadata(
            coordinateTransformations=self.transformations,
            coordinateSystems=self.metadata.coordinateSystems,
        )
