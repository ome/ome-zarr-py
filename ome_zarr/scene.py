# the class for storage representation, not exposed to the user
from dataclasses import field, dataclass

from ome_zarr_models._v06.coordinate_transforms import (
  Transform,
  CoordinateSystemIdentifier
)

from .image import NgffMultiscales
import zarr
import networkx as nx

@dataclass
class SceneMetadata:
  coordinateTransformations: list[Transform]
  coordinateSystems: tuple[CoordinateSystemIdentifier] | list[CoordinateSystemIdentifier] = field(default_factory=list)
  


# the class exposed to the user
@dataclass(kw_only=True)
class NgffScene:
    images: list[NgffMultiscales]
    transformations: list[Transform]
    coordinate_systems: tuple[CoordinateSystemIdentifier] | list[CoordinateSystemIdentifier] = field(default_factory=list)

    def __post_init__(self):

        self.metadata = SceneMetadata(
          coordinateTransformations=self.transformations,
          coordinateSystems=self.coordinate_systems
          )
        
        self._graph = nx.DiGraph()

        for cs in self.coordinate_systems:
            if cs.path is None:
                self._graph.add_node((None, cs.name))

        for img in self.images:
            # add all coordinate systems as nodes
            for cs in img.metadata.coordinateSystems:
                node_id = (img.name, cs.name)
                self._graph.add_node(node_id)

            for ds in img.metadata.datasets:
                # add all datasets as nodes
                node_id = (img.name, ds.path)
                self._graph.add_node(node_id)

                # add scale transformations from dataset as edges
                transform = ds.coordinateTransformations
                self._graph.add_edge(
                    (img.name, ds.path),
                    (img.name, ds.coordinateTransformations[0].output),
                    transform=transform)
                
            # add additional transformations from image metadata as edges
            if img.metadata.coordinateTransformations:
                for tf in img.metadata.coordinateTransformations:
                    self._graph.add_edge(
                        (img.name, tf.input),
                        (img.name, tf.output),
                        transform=tf)
        
        # add scene-level transformations as edges between coordinate systems of different images
        for tf in self.transformations:
            self._graph.add_edge(
                (tf.input.path, tf.input.name),
                (tf.output.path, tf.output.name),
                transform=tf)

    def to_ome_zarr(self, path):
        # create zarr group and store the scene metadata
        zarr_group = zarr.open(path, mode='w')
        
        # Create a subgroup for each image using its name
        for i, img in enumerate(self.images):
            subgroup = zarr_group.create_group(img.metadata.name)
            img.to_ome_zarr(subgroup)

        metadata_dict = {
            "coordinateTransformations": [t.model_dump() for t in self.metadata.coordinateTransformations],
            "coordinateSystems": [cs.model_dump() for cs in self.metadata.coordinateSystems] if self.metadata.coordinateSystems else None
        }

        zarr_group.attrs['ome'] = {
            "scene": metadata_dict,
            "version": "0.6"
        }