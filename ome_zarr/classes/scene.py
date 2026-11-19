# the class for storage representation, not exposed to the user
from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from functools import partial
from typing import Any, cast

import transformnd as tnd
import zarr
from ome_zarr_models.v06.coordinate_transforms import (
    AnyTransform,
    CoordinateSystem,
)
from ome_zarr_models.v06.multiscales import Multiscale
from ome_zarr_models.v06.scene import BaseSceneAttrs, SceneAttrs
from pydantic import TypeAdapter, ValidationError
from zarr.storage import StoreLike

from .image import OMEZarrMultiscale
from .tnd_utils import (
    UnsupportedTransformation,
    ozmp_tf_to_tnd_spaced,
)

LOGGER = logging.getLogger(__name__)
TRANSFORM_ADAPTER = TypeAdapter(AnyTransform)

VECTOR_FIELD_AXIS = 0
"""Which dimension of a vector field array contains the vectors."""


def _read_scene_metadata(
    attributes: Any, *, strict: bool
) -> tuple[SceneAttrs, tuple[Any, ...]]:
    try:
        return BaseSceneAttrs.model_validate(attributes).scene, ()
    except ValidationError as error:
        if strict:
            raise

        invalid: dict[int, str] = {}
        for item in error.errors(include_input=False, include_url=False):
            location = item["loc"]
            if (
                len(location) < 3
                or location[:2] != ("scene", "coordinateTransformations")
                or not isinstance(location[2], int)
            ):
                raise
            invalid.setdefault(location[2], item["msg"])

        scene = attributes["scene"]
        transforms = scene["coordinateTransformations"]
        valid = [tf for i, tf in enumerate(transforms) if i not in invalid]
        metadata = BaseSceneAttrs.model_validate(
            {**attributes, "scene": {**scene, "coordinateTransformations": valid}}
        )
        for i, message in invalid.items():
            LOGGER.warning("Skipping invalid scene transformation %d (%s)", i, message)
        return metadata.scene, tuple(transforms[i] for i in invalid)


def _read_image_metadata(
    attributes: dict[str, Any], *, path: str, unparsed_images: set[str]
) -> Multiscale:
    transforms = attributes.get("coordinateTransformations")
    if not isinstance(transforms, list | tuple):
        return Multiscale.model_validate(attributes)

    # The image model checks transform dimensionality while parsing. Defer
    # those checks for optional edges to the scene graph, where unavailable
    # transforms can be skipped without discarding the image.
    metadata = Multiscale.model_validate(
        {**attributes, "coordinateTransformations": None}
    )
    valid = []
    for i, transform in enumerate(transforms):
        try:
            valid.append(TRANSFORM_ADAPTER.validate_python(transform))
        except ValidationError as error:
            message = error.errors(include_input=False, include_url=False)[0]["msg"]
            LOGGER.warning(
                "Skipping invalid transformation %d in %s (%s)", i, path, message
            )
            unparsed_images.add(path)
    return metadata.model_copy(update={"coordinateTransformations": tuple(valid)})


class OMEZarrScene:
    def __init__(
        self,
        images: list[OMEZarrMultiscale] | dict[str, OMEZarrMultiscale],
        coordinate_transformations: Sequence[AnyTransform],
        coordinate_systems: Sequence[CoordinateSystem] = (),
        coordinate_displacements: dict[str, OMEZarrMultiscale] | None = None,
        *,
        strict: bool = True,
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
        strict : bool, optional
            Require every transformation to be usable when building the graph.
            With False, unusable transformations are logged and skipped.
            Writing a scene always requires strict validation.

        Methods
        -------
        to_ome_zarr(store: StoreLike, overwrite: bool = False, compute: bool = True)
            Write the scene to OME-Zarr format in the specified store.
        from_ome_zarr(store: StoreLike, *, strict: bool = False)
            Load an existing scene from OME-Zarr format in the specified store.
        get_coordinate_system(path: str | None, name: str | None = None)
            Retrieve coordinate systems matching the path (if given) and name (if given).

        """
        # Coerce list to dict keyed by metadata.name
        if isinstance(images, list):
            self.images = {str(img.metadata.name): img for img in images}
        else:
            self.images = images

        if coordinate_displacements is None:
            coordinate_displacements = dict()

        self.coordinate_displacements = coordinate_displacements

        # metadata is the single source of truth
        self._metadata = SceneAttrs(
            coordinateSystems=tuple(coordinate_systems),
            coordinateTransformations=tuple(coordinate_transformations),
        )

        self._unparsed_transformations: tuple[Any, ...] = ()
        self._images_with_unparsed_transforms: dict[str, OMEZarrMultiscale] = {}
        self._graph = self._setup_graph(strict=strict)

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
        self._metadata = SceneAttrs.model_validate(
            {
                "coordinateSystems": self._metadata.coordinateSystems,
                "coordinateTransformations": value,
            }
        )
        self._unparsed_transformations = ()

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
        previous = self._metadata
        self._metadata = value
        try:
            graph = self._setup_graph()
        except Exception:
            self._metadata = previous
            raise
        self._graph = graph
        self._unparsed_transformations = ()

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

    def _add_transform(
        self,
        graph: tnd.TransformGraph,
        transform: AnyTransform,
        *,
        strict: bool,
        subgroup: str | None = None,
    ) -> None:
        try:
            source, target = transform.input, transform.output
            if source is None or target is None:
                raise ValueError(
                    f"Coordinate transformation {transform} is missing input or output information."
                )

            if subgroup is None:
                if source.name is None or target.name is None:
                    raise UnsupportedTransformation(
                        "Scene coordinate systems must be named", transform
                    )
                source_cs = self.get_coordinate_system(source.path, source.name).get(
                    (source.path or "", source.name)
                )
                target_cs = self.get_coordinate_system(target.path, target.name).get(
                    (target.path or "", target.name)
                )
                if source_cs is None or target_cs is None:
                    unknown = source if source_cs is None else target
                    raise UnsupportedTransformation(
                        f"Unknown coordinate system {unknown}", transform
                    )
            else:
                image = self.images[subgroup]
                source_cs = image._get_descendant_coordinate_system(source)
                target_cs = image._get_descendant_coordinate_system(target)

            spaced = ozmp_tf_to_tnd_spaced(
                transform,
                zarr_context=subgroup or "",
                source_cs=source_cs,
                target_cs=target_cs,
                coordinate_displacements=self.coordinate_displacements,
            )
            inverse = spaced.invert()

            # Check both spaces before insertion. TransformGraph adds its
            # source node before checking the target dimension.
            spaces, dims = spaced.spaces, spaced.transform.ndims
            if spaces.source == spaces.target and dims.source != dims.target:
                raise ValueError(f"Inconsistent dimensions for space {spaces.source}")
            endpoints = (
                (spaces.source, dims.source, source_cs),
                (spaces.target, dims.target, target_cs),
            )
            for space, ndim, coordinate_system in endpoints:
                if (
                    coordinate_system is not None
                    and len(coordinate_system.axes) != ndim
                ):
                    raise ValueError(
                        f"Coordinate system {space} has {len(coordinate_system.axes)} axes, got {ndim}D"
                    )
                known = graph.ndim(space)
                if known is not None and ndim != known:
                    raise ValueError(f"Space {space} is {known}D, got {ndim}D")
        except (UnsupportedTransformation, ValueError) as error:
            if strict:
                raise
            LOGGER.warning(
                "Skipping %s transformation in %s (%s)",
                transform.type,
                subgroup or "the scene",
                error,
            )
            return

        graph.add_transform(spaced)
        if inverse is not None:
            graph.add_transform(inverse)

    def _setup_graph(self, *, strict: bool = True) -> tnd.TransformGraph:
        g = tnd.TransformGraph()
        for tf in self.coordinate_transformations:
            self._add_transform(g, tf, strict=strict)
        for subgroup, image in self.images.items():
            for tf in image.metadata.coordinateTransformations or ():
                self._add_transform(g, tf, strict=strict, subgroup=subgroup)
        return g

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

        if self._unparsed_transformations:
            raise ValueError(
                "The scene contains invalid coordinate transformation metadata. "
                "Replace coordinate_transformations before writing."
            )
        for path, image in self._images_with_unparsed_transforms.items():
            if self.images.get(path) is image:
                raise ValueError(
                    f"Image {path!r} contains invalid coordinate transformation metadata. "
                    "Replace the image before writing."
                )
        self._setup_graph()

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
    def from_ome_zarr(cls, store: StoreLike | zarr.Group, *, strict: bool = False):
        """
        Load an existing scene from OME-Zarr format.

        By default, invalid or unusable coordinate transformations are logged
        and skipped when building the graph. Use ``strict=True`` to require
        every transformation to be valid and usable. Correct or replace
        invalid transformations before writing a scene.
        """
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
        scene_metadata, unparsed = _read_scene_metadata(
            zarr_group.attrs.get("ome", {}), strict=strict
        )
        transformations = scene_metadata.coordinateTransformations
        coordinate_systems = scene_metadata.coordinateSystems

        # Load all image subgroups, keyed by their zarr path
        paths = dict.fromkeys(
            endpoint.path
            for tf in transformations
            for endpoint in (tf.input, tf.output)
            if endpoint is not None and endpoint.path is not None
        )
        # A malformed transform can still identify an image in the scene.
        for tf in unparsed:
            if not isinstance(tf, dict):
                continue
            for endpoint in (tf.get("input"), tf.get("output")):
                if isinstance(endpoint, dict) and isinstance(endpoint.get("path"), str):
                    paths.setdefault(endpoint["path"], None)

        images = {}
        unparsed_images: set[str] = set()
        for path in paths:
            img_group = zarr_group.get(path)
            if not isinstance(img_group, zarr.Group):
                if strict:
                    raise ValueError(
                        f"Image specified in metadata at '{path}' not found in zarr group."
                    )
                continue
            if strict:
                images[path] = OMEZarrMultiscale.from_ome_zarr(img_group)
            else:
                images[path] = cast(
                    OMEZarrMultiscale,
                    OMEZarrMultiscale._load_from_ome_zarr(
                        img_group,
                        parse_multiscale=partial(
                            _read_image_metadata,
                            path=path,
                            unparsed_images=unparsed_images,
                        ),
                    ),
                )

        scene = cls(
            images=images,
            coordinate_transformations=transformations,
            coordinate_systems=(
                coordinate_systems if coordinate_systems is not None else ()
            ),
            coordinate_displacements=coordinate_displacements,
            strict=strict,
        )
        scene._unparsed_transformations = unparsed
        scene._images_with_unparsed_transforms = {
            path: images[path] for path in unparsed_images
        }
        return scene
