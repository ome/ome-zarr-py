# the class for storage representation, not exposed to the user
import os
import posixpath
from collections.abc import Sequence
from typing import Any, cast

import transformnd as tnd
import zarr
from ome_zarr_models.v06.coordinate_transforms import (
    AnyTransform,
    CoordinateSystem,
)
from ome_zarr_models.v06.scene import SceneAttrs
from zarr.storage import StoreLike

from .image import OMEZarrMultiscale


class OMEZarrScene:
    def __init__(
        self,
        images: list[OMEZarrMultiscale] | dict[str, OMEZarrMultiscale],
        coordinate_transformations: Sequence[AnyTransform],
        coordinate_systems: Sequence[CoordinateSystem] = (),
        coordinate_displacements: dict[str, OMEZarrMultiscale] | None = None,
    ):
        """
        Parameters
        ----------
        images : list[OMEZarrMultiscale] | dict[str, OMEZarrMultiscale]
            Either a list of images (keyed internally by metadata.name) or a dict
            mapping zarr group paths to images. The dict form gives explicit control
            over the paths where images will be stored in the zarr hierarchy.
        coordinate_transformations : Sequence[ome_zarr_models.v06.coordinate_transforms.AnyTransform]
            A sequence of coordinate transformations that define how to map between
            different coordinate systems in the scene.
            Each transformation can be provided as an AnyTransform instance
            or as a dictionary that can be validated into an AnyTransform.
            For more information see [ngff specification](https://ngff.openmicroscopy.org/specifications/dev/index.html#coordinatetransformations-metadata).
        coordinate_systems : Sequence[ome_zarr_models.v06.coordinate_transforms.CoordinateSystem]
            A sequence of coordinate systems that define the coordinate spaces used
            in the scene.
            Each coordinate system can be provided as a CoordinateSystem instance
            or as a dictionary that can be validated into a CoordinateSystem.
            For more information see [ngff specification](https://ngff.openmicroscopy.org/specifications/dev/index.html#coordinatesystems-metadata).
        coordinate_displacements : dict[str, OMEZarrMultiscale] | None, optional
            A dictionary mapping zarr group paths to displacement field images
            or coordinate arrays, which are referenced by the [displacements and
            coordinates transformations](https://ngff.openmicroscopy.org/specifications/dev/index.html#coordinates-and-displacements) in the scene metadata.

        Methods
        -------
        to_ome_zarr(store: StoreLike, overwrite: bool = False, compute: bool = True)
            Write the scene to OME-Zarr format in the specified store.
        from_ome_zarr(store: StoreLike)
            Load an existing scene from OME-Zarr format in the specified store.
        get_coordinate_system(path: str | None, name: str | None = None)
            Retrieve a coordinate system by path or name.
            If neither is


        """
        # Coerce list to dict keyed by metadata.name
        if isinstance(images, list):
            self.images = {str(img.metadata.name): img for img in images}
        else:
            self.images = images

        self.coordinate_displacements = coordinate_displacements

        # metadata is the single source of truth
        self._metadata = SceneAttrs(
            coordinateSystems=tuple(coordinate_systems),
            coordinateTransformations=tuple(coordinate_transformations),
        )

        self._build_graph()

    @property
    def coordinate_systems(self) -> tuple[CoordinateSystem, ...]:
        return tuple(self._metadata.coordinateSystems or ())

    @coordinate_systems.setter
    def coordinate_systems(self, value: Sequence[CoordinateSystem]) -> None:
        self._metadata = self._metadata.model_copy(update={"coordinateSystems": value})

    @property
    def coordinate_transformations(self) -> tuple[AnyTransform, ...]:
        return tuple(self._metadata.coordinateTransformations or ())

    @coordinate_transformations.setter
    def coordinate_transformations(
        self, value: Sequence[AnyTransform] | list[dict[str, Any]]
    ) -> None:
        self._metadata = self._metadata.model_copy(
            update={"coordinateTransformations": value}
        )

    @property
    def metadata(self) -> SceneAttrs:
        """
        Get the scene metadata as a SceneAttrs object.

        Returns
        -------
        SceneAttrs: ome_zarr_models.v06.scene.SceneAttrs
            The scene metadata, including coordinate systems and transformations.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, value: SceneAttrs) -> None:
        self._metadata = value
        self._build_graph()

    def get_coordinate_system(
        self, path: str | None = None, name: str | None = None
    ) -> dict[tuple[str, str], CoordinateSystem]:
        """
        Retrieve coordinate systems.

        Parameters
        ----------
        name: str | None
            Optional name of the coordinate system to retrieve.
            If provided, only coordinate systems with this name will be returned.
        path: str | None
            Optional path to disambiguate coordinate systems by the zarr path
            under which they are stored.
            If provided, only coordinate systems associated with this path will be returned.

        Returns
        -------
        dict[tuple[str, str], CoordinateSystem]
            A dictionary of matching CoordinateSystem objects keyed by (path, name) tuples.
            Empty if no match is found.
        """
        matches = {}

        # Add top-level coordinate systems
        for cs in self.coordinate_systems:
            if (name is None or cs.name == name) and (path is None or path == ""):
                matches[("", cs.name)] = cs

        # Add image-level coordinate systems
        for img_path, img in self.images.items():
            for cs in img.metadata.coordinateSystems:
                if (name is None or cs.name == name) and (
                    path is None or path == img_path
                ):
                    matches[(img_path, cs.name)] = cs

        return matches

    def _build_graph(self):
        self._graph = tnd.graph.TransformGraph()
        # Add scene-level transformations (empty context = root level)
        for tf in self.coordinate_transformations:
            if tf.input is None or tf.output is None:
                raise ValueError(
                    f"Coordinate transformation {tf} is missing input or output information."
                )
            source_cs_dict = self.get_coordinate_system(tf.input.path, tf.input.name)
            source_cs = source_cs_dict[(tf.input.path or "", tf.input.name)]

            # convert to transformnd transform and add to graph
            target_cs_dict = self.get_coordinate_system(tf.output.path, tf.output.name)
            target_cs = target_cs_dict[(tf.output.path or "", tf.output.name)]
            tnd_transform = self._ozmp_tf_to_tnd(
                tf,
                zarr_context="",
                source_cs=source_cs,
                target_cs=target_cs,
            )
            self._graph.add_transform(tnd_transform)

            # Add inverse edge if transform is invertible
            inverse = tnd_transform.invert()
            if inverse is not None:
                self._graph.add_transform(inverse)

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
                        ind_transform = self._ozmp_tf_to_tnd(
                            img_tf,
                            zarr_context=subgroup,
                            source_cs=None,
                            target_cs=None,
                        )
                        self._graph.add_transform(ind_transform)
                        # Add inverse edge if transform is invertible
                        inverse = ind_transform.invert()
                        if inverse is not None:
                            self._graph.add_transform(inverse)

    def to_ome_zarr(
        self, store: StoreLike, overwrite: bool = False, compute: bool = True
    ) -> list:
        """
        Write scene to OME-Zarr format.

        Parameters
        ----------
        store: StoreLike
            A zarr-compatible storage backend (e.g., directory path, in-memory store, etc.)
        overwrite: bool
            If True, overwrite all images in the store with the current state of the scene.
            If False, only write new images that haven't been written before. Existing images in the store will be left unchanged.
        compute: bool
            If True, compute any lazy operations on write of each image.
            Otherwise return a list of delayed operations that can be computed later.

        """
        import shutil

        if overwrite and os.path.exists(str(store)):
            # Clear the store if it already exists and we're not doing incremental writes
            shutil.rmtree(str(store))

        # Open or create zarr group
        mode = "w" if overwrite else "a"
        zarr_group = zarr.open(store, mode=mode)

        delayed = []

        # Create a subgroup for each image using its path key
        for img_path, img in self.images.items():
            # Skip if already written (incremental mode)
            if not overwrite and img_path in zarr_group:
                continue

            # Write the image
            subgroup = zarr_group.create_group(img_path, overwrite=overwrite)
            delayed += img.to_ome_zarr(
                subgroup, overwrite=True, version="0.6", compute=compute
            )

        for disp_path, disp_img in (self.coordinate_displacements or {}).items():
            # Skip if already written (incremental mode)
            if not overwrite and disp_path in zarr_group:
                continue

            # Write the displacement image
            subgroup = zarr_group.create_group(
                f"coordinateTransformations/{disp_path}", overwrite=overwrite
            )
            delayed += disp_img.to_ome_zarr(
                subgroup, overwrite=True, version="0.6", compute=compute
            )

        # Always update scene metadata
        metadata_dict = self.metadata.model_dump(exclude_none=True)

        zarr_group.attrs["ome"] = {"scene": metadata_dict, "version": "0.6"}

        return delayed

    @classmethod
    def from_ome_zarr(cls, store: StoreLike):
        """
        Load an existing scene from OME-Zarr format.

        Args:
            path: Path to the OME-Zarr scene

        Returns:
            NgffScene instance with images and metadata loaded from disk
        """
        from ome_zarr_models.v06.scene import BaseSceneAttrs

        # Handle both StoreLike (string, dict, etc.) and zarr.Group objects
        if isinstance(store, zarr.Group):
            zarr_group = store
        else:
            zarr_group = zarr.open(store, mode="r")

        # load coordinateTransformations array data, if it exists
        if "coordinateTransformations" in zarr_group:
            coordinate_displacements = {}
            for disp_path in zarr_group["coordinateTransformations"].group_keys():
                disp_group = zarr_group["coordinateTransformations"][disp_path]
                disp_img = cast(
                    OMEZarrMultiscale, OMEZarrMultiscale.from_ome_zarr(disp_group)
                )
                coordinate_displacements[disp_path] = disp_img
        else:
            coordinate_displacements = None

        # Load scene metadata
        scene_metadata = BaseSceneAttrs.model_validate(zarr_group.attrs.get("ome", {}))
        transformations = scene_metadata.scene.coordinateTransformations
        coordinate_systems = scene_metadata.scene.coordinateSystems

        # Load all image subgroups, keyed by their zarr path
        images = {}
        for tf in transformations:
            if tf.input is not None:
                path = tf.input.path
                if path is not None and path in zarr_group:
                    img_group = zarr_group[path]
                    img = cast(
                        OMEZarrMultiscale, OMEZarrMultiscale.from_ome_zarr(img_group)
                    )
                    images[path] = img
                elif path is not None and path not in zarr_group:
                    raise ValueError(
                        f"Image specified in metadata at '{path}' not found in zarr group."
                    )
            if tf.output is not None:
                path = tf.output.path
                if path is not None and path in zarr_group:
                    img_group = zarr_group[path]
                    img = cast(
                        OMEZarrMultiscale, OMEZarrMultiscale.from_ome_zarr(img_group)
                    )
                    images[path] = img
                elif path is not None and path not in zarr_group:
                    raise ValueError(
                        f"Image specified in metadata at '{path}' not found in zarr group."
                    )

        scene = OMEZarrScene(
            images=images,
            coordinate_transformations=transformations,
            coordinate_systems=(
                coordinate_systems if coordinate_systems is not None else ()
            ),
            coordinate_displacements=coordinate_displacements,
        )

        return scene

    def _ozmp_tf_to_tnd(
        self,
        transform: AnyTransform,
        zarr_context: str = "",
        source_cs: CoordinateSystem | None = None,
        target_cs: CoordinateSystem | None = None,
    ) -> tnd.base.Transform:
        """
        Convert an OME-Zarr coordinate transformation to a transformnd Transform object.
        This is a placeholder function and will need to be implemented based on the specific types of transformations you expect to encounter in OME-Zarr metadata.
        """
        import numpy as np

        if transform.input is not None:
            input_path = transform.input.path or ""
            output_path = transform.output.path or ""

            # zarr_context prepends path with relative path from root
            # to keep track of global location of coordinate systems in the zarr store
            if zarr_context:
                input_path = (
                    posixpath.join(zarr_context, input_path)
                    if input_path
                    else zarr_context
                )
                output_path = (
                    posixpath.join(zarr_context, output_path)
                    if output_path
                    else zarr_context
                )

            spaces = tnd.Spaces(
                (input_path, transform.input.name),
                (output_path, transform.output.name),
            )
        else:
            spaces = tnd.Spaces(None, None)

        tnd_transform = None
        # Example for an affine transformation (this will depend on the actual structure of AnyTransform)
        if transform.type == "affine":
            aff = np.asarray(transform.affine)
            if aff.shape[0] == aff.shape[1]:
                tnd_transform = tnd.transforms.Affine(
                    transform.affine,
                    spaces=spaces,
                )
            else:
                aff = np.eye(max(aff.shape))
                aff[: aff.shape[0], : aff.shape[1]] = aff
                tnd_transform = tnd.transforms.Affine(aff, spaces=spaces)

        elif transform.type == "displacements":
            path_to_dfield = transform.path or ""
            if zarr_context and path_to_dfield:
                path_to_dfield = posixpath.join(zarr_context, path_to_dfield)

            if self.coordinate_displacements is not None:
                dfield = self.coordinate_displacements.get(
                    posixpath.basename(path_to_dfield)
                )
                if dfield is not None:
                    if dfield.images[0].scale is None:
                        raise ValueError(
                            f"Displacement field at {path_to_dfield} is missing scale information."
                        )
                    tnd_transform = tnd.transforms.Displacements(
                        dfield.images[0].data,
                        index_transform=tnd.transforms.Scale(
                            list(dfield.images[0].scale.values())[1:]
                        ),
                        vector_axis=0,
                        spaces=spaces,
                    )
        elif transform.type == "mapAxis":
            tnd_transform = tnd.transforms.MapAxis(
                list(transform.mapAxis),
                spaces=spaces,
            )

        elif transform.type == "projectAxis":
            tnd_transform = tnd.transforms.ProjectAxis(
                created=transform.createdOutputs,
                dropped=transform.droppedInputs,
                spaces=spaces,
                source_ndim=len(source_cs.axes) if source_cs is not None else None,
                target_ndim=len(target_cs.axes) if target_cs is not None else None,
            )

        elif transform.type == "scale":
            tnd_transform = tnd.transforms.Scale(transform.scale, spaces=spaces)

        elif transform.type == "translation":
            tnd_transform = tnd.transforms.Translate(
                transform.translation, spaces=spaces
            )

        elif transform.type == "rotation":
            affine_matrix = np.eye(len(transform.rotation) + 1)
            affine_matrix[:-1, :-1] = transform.rotation
            tnd_transform = tnd.transforms.Affine(affine_matrix, spaces=spaces)

        elif transform.type == "byDimension":
            sub_transformations = transform.transformations
            tnd_sub_transforms = [
                tnd.transforms.by_dimension.SubTransform(
                    transform=self._ozmp_tf_to_tnd(sub_tf.transformation),
                    input_axes=sub_tf.inputAxes,
                    output_axes=sub_tf.outputAxes,
                )
                for sub_tf in sub_transformations
            ]
            tnd_transform = tnd.transforms.ByDimension(
                subtransforms=tnd_sub_transforms,
                fill_identity=0,
                spaces=spaces,
            )
        elif transform.type == "sequence":
            sub_transformations = transform.transformations
            tnd_sub_transforms = [
                self._ozmp_tf_to_tnd(sub_tf, zarr_context, source_cs, target_cs)
                for sub_tf in sub_transformations
            ]
            tnd_transform = tnd.base.TransformSequence(
                tnd_sub_transforms,
                spaces=spaces,
            )

        return tnd_transform
