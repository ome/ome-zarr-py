from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field
from typing import Any, cast
import typing

import dask.array as da
import numpy as np
import zarr
from ome_zarr_models.common.image_label_types import LabelBase as Label
from ome_zarr_models.common.omero import Omero

from ome_zarr_models._v06.coordinate_transforms import (
    Scale,
    Translation,
    Identity,
    Sequence as TransformSequence,
    Axis,
    CoordinateSystem,
    CoordinateSystemIdentifier,
    AnyTransform,
)
from ome_zarr_models._v06.multiscales import (
    Dataset,
    Multiscale as MultiscaleV06,
)
from ome_zarr_models.v05.multiscales import (
    Multiscale as MultiscaleV05,
)
from pydantic import ValidationError, TypeAdapter

from ome_zarr.scale import Methods

SPATIAL_DIMS = ["z", "y", "x"]


@dataclass
class NgffImage:
    """
    Single-scale image representation with metadata.

    Parameters
    ----------
    data : dask.array.Array or numpy.ndarray
        The image data array.
    axes : sequence of str or str
        The axis names corresponding to the data array axes, i.e. ('c', 'z', 'y', 'x').
    scale : sequence of float or dict of str to float, optional
        The physical scale for each axis. If a sequence is provided, it should
        match the order of `axes`. If a dict is provided, keys should be axis names,
        e.g. {'x': 0.1, 'y': 0.1, 'z': 0.5}. Default is None, which sets all scales to 1.0.
    axes_units : dict of str to str, optional
        Units for each axis, e.g. {'x': 'micrometer', 'y': 'micrometer'}.
        Default is None (no units).
    name : str, optional
        Name of the image. Default is "image".

    Attributes
    ----------
    data : dask.array.Array
        The image data array.
    axes : sequence of str
        The axis names.
    scale : dict of str to float
        The physical scale for each axis.
    axes_units : dict of str to str
        Units for each axis, e.g. {'x': 'micrometer', 'y': 'micrometer'}.
        Default is None (no units).
    name : str
        Name of the image.
    """

    data: da.Array | np.ndarray
    axes: Sequence[str] | str
    scale: Sequence[float] | dict[str, float] | None = None
    axes_units: dict[str, str] | None = None
    name: str | None = "image"

    def __post_init__(self):
        # set default scale if unset
        if self.scale is None:
            self.scale = tuple(1.0 for _ in range(len(self.axes)))

        # coerce axes to list
        if isinstance(self.axes, str):
            self.axes = list(self.axes)

        # coerce scale to dict if it's a sequence
        if isinstance(self.scale, Sequence):
            if len(self.scale) != len(self.axes):
                raise ValueError(
                    f"Number of scale values ({len(self.scale)}) "
                    f"does not match number of axes ({len(self.axes)})"
                )
            self.scale = dict(zip(self.axes, self.scale))

        # coerce data to dask array
        if not isinstance(self.data, da.Array):
            self.data = da.from_array(self.data)

        # validate dimensions match data shape
        if len(self.axes) != len(self.data.shape):
            raise ValueError(
                f"Number of dimensions in data ({len(self.data.shape)}) "
                f"does not match number of dims ({len(self.axes)})"
            )


@dataclass
class NgffMultiscales:
    """
    Container for multiscale image pyramid with OME-Zarr metadata.

    Parameters
    ----------
    image : NgffImage
        The base (highest resolution) image.
    scale_factors : list of int, optional
        Downsampling factors for each pyramid level.
        If passed as a list of integers (i.e. [2, 4, 8]),
        the same factors will be applied to all *spatial* dimensions
        except for the z-axis (if present).
        To customize this behavior, pass a list of dicts mapping dimension names to factors, e.g.
        `[{'x': 2, 'y': 2}, {'x': 4, 'y': 4}, {'x': 8, 'y': 8}]`
        Default: [2, 4, 8, 16].
    method : str or Methods, optional
        Downsampling method to use. Default: Methods.RESIZE.
    labels : :class:`NgffMultiscales` or dict of str to :class:`NgffMultiscales`, optional
        Optional labels to associate with the image pyramid.
        Can be a single :class:`NgffMultiscales` instance (for a single label pyramid)
        or a dict mapping label names to :class:`NgffMultiscales` instances
        (for multiple label pyramids), e.g.
        `{'nuclei': nuclei_multiscale, 'cells': cells_multiscale}`.
        Default is None (no labels).
    omero : dict or Omero, optional
        Optional Omero metadata to include in the OME-Zarr attributes.
        Can be passed as a dict or an instance of the [Omero model](https://ome-zarr-models-py.readthedocs.io/en/stable/api/v04/image/#omero-metadata)
        Default is None (no Omero metadata).
        For example metadata, see [ngff specification](https://ngff.openmicroscopy.org/specifications/0.5/index.html#omero-metadata-transitional)
    image_label : dict or Label, optional
        Optional image-label metadata to describe rendering options specifically for label images.
        Can signal to viewers that this image should be rendered as labels.
        Can be passed as a dict or an instance of the [Label model](https://ome-zarr-models-py.readthedocs.io/en/stable/api/v05/image-label/)
        For example metadata, see [ngff specification](https://ngff.openmicroscopy.org/specifications/0.5/index.html#labels-metadata)

    Attributes
    ----------
    images : list of NgffImage
        List of images at each pyramid level.
    metadata : Multiscale
        OME-Zarr multiscale metadata.
    omero : Omero or None
        Optional Omero metadata included in the OME-Zarr attributes.
    image_label : Label or None
        Optional image-label metadata included in the OME-Zarr attributes.
    """

    # Attributes that are populated in __post_init__ and not passed by the user
    images: list[NgffImage] = field(init=False)

    # InitVars for parameters passed by the user
    image: InitVar[NgffImage]
    scale_factors: InitVar[
        list[int] | tuple[int, ...] | list[dict[str, int]] | None
    ] = None
    method: str | Methods | None = Methods.RESIZE
    coordinateTransformations: InitVar[tuple[AnyTransform, ...] | list[dict[str, Any]] | None] = (
        None
    )
    labels: (
        NgffMultiscales | list[NgffMultiscales] | dict[str, NgffMultiscales] | None
    ) = None
    omero: dict[str, Any] | Omero | None = None
    image_label: dict[str, Any] | Label | None = None
    default_coordinate_system_name: str = "physical"

    def __post_init__(
        self,
        image: NgffImage,
        scale_factors: list[int] | tuple[int, ...] | list[dict[str, int]] | None,
        coordinateTransformations: tuple[AnyTransform, ...] | list[dict[str, Any]] | None,
    ):
        from ome_zarr.scale import _build_pyramid

        if scale_factors is None:
            scale_factors = (2, 4, 8, 16)

        self.name = image.name
        method = self.method

        if isinstance(method, Methods):
            method = str(method.value)
        elif method is None:
            method = str(Methods.RESIZE.value)

        if self.image_label is not None and method != str(Methods.NEAREST.value):
            warnings.warn(
                f"Image label metadata provided but method is {method}, not 'nearest'. "
                f"Consider using method='nearest' label image multiresolution pyramids."
            )

        # Build the pyramid data
        pyramid = _build_pyramid(
            image=image.data,
            dims=image.axes,
            scale_factors=scale_factors,
            method=method,
        )

        # build scales for each level based on the original image shape
        # and the pyramid level shapes
        scales = []
        # image.scale is guaranteed to be a dict after NgffImage.__post_init__
        image_scale = image.scale
        assert isinstance(image_scale, dict)
        for shape in [d.shape for d in pyramid]:
            scale = [full / level for full, level in zip(image.data.shape, shape)]
            scales.append(
                {
                    d: s * image_scale[d] if d in image_scale else 1.0
                    for d, s in zip(image.axes, scale)
                }
            )

        # Create Image instances for each pyramid level
        images = []
        datasets = []
        for idx, (level_data, level_scale) in enumerate(zip(pyramid, scales)):

            images.append(
                NgffImage(
                    data=level_data,
                    axes=image.axes,
                    scale=level_scale,
                    axes_units=image.axes_units,
                    name=image.name,
                )
            )
            datasets.append(
                Dataset(
                    path=f"s{idx}",
                    coordinateTransformations=(
                        Scale(
                            type="scale",
                            scale=tuple(level_scale.values()),
                            input=CoordinateSystemIdentifier(
                                path=f"s{idx}"
                            ),
                            output=CoordinateSystemIdentifier(
                                name=self.default_coordinate_system_name
                            )
                        ),
                    ),
                )
            )

        self.images = images

        # Build axes metadata
        if image.axes_units is None:
            image.axes_units = {}

        axes = []
        for d in image.axes:
            if d in SPATIAL_DIMS:
                axes.append(Axis(name=d, type="space", unit=image.axes_units.get(d)))
            elif d == "t":
                axes.append(Axis(name=d, type="time", unit=image.axes_units.get(d)))
            elif d == "c":
                axes.append(Axis(name=d, type="channel", unit=image.axes_units.get(d)))
            else:
                axes.append(Axis(name=d, type="custom", unit=image.axes_units.get(d)))

        coordinate_system = CoordinateSystem(
            axes=tuple(axes),
            name=self.default_coordinate_system_name,
        )

        # coerce coordinateTransformations to ozmp object if passed as dict
        transforms = None
        if coordinateTransformations is not None:
            transforms = []
            for idx, tf in enumerate(coordinateTransformations):
                if isinstance(tf, dict):
                    transforms.append(TypeAdapter(AnyTransform).validate_python(tf))
                elif tf is AnyTransform:
                    transforms.append(tf)
                else:
                    raise ValueError(
                        f"Invalid coordinate transformation at index {idx}: {tf}"
                    )
            transforms = tuple(transforms)

            # Some checks on the transform's input and output coordinate system
            for idx, tf in enumerate(transforms):

                # first, check that they are not None
                if tf.input is None:
                    raise ValueError(
                        f"Coordinate transformation at index {idx} is missing input coordinate system: {tf}"
                    )
                if tf.output is None:
                    raise ValueError(
                        f"Coordinate transformation at index {idx} is missing output coordinate system: {tf}"
                    )

                # now check that either input or output name matches the default coordinate system name
                if (
                    tf.input.name != self.default_coordinate_system_name
                    and tf.output.name != self.default_coordinate_system_name
                ):
                    raise ValueError(
                        f"Coordinate transformation at index {idx} must have either "
                        f"input or output coordinate system name matching the default "
                        f"coordinate system name '{self.default_coordinate_system_name}': {tf}"
                    )

        self.metadata = MultiscaleV06(
            coordinateSystems=tuple([coordinate_system]),
            datasets=tuple(datasets),
            name=image.name,
            coordinateTransformations=transforms,
        )

        # coerce labels to dict if it's a single NgffMultiscales or a list
        self.labels = self._parse_labels(self.labels)
        self.omero = self._parse_omero(self.omero)
        self.image_label = self._parse_image_label(self.image_label)

    def to_ome_zarr(
        self,
        group: zarr.Group | str,
        storage_options: list[dict[str, Any]] | dict[str, Any] | None = None,
        version: str | None = "0.6",
        compute: bool = True,
        overwrite: bool = True,
    ) -> list:
        """
        Serialize the multiscale pyramid to an OME-Zarr group.

        Parameters
        ----------
        group : zarr.Group or str
            The target Zarr group or path where the OME-Zarr data will be written.
        storage_options : dict or list of dict, optional
            Additional storage options to pass to Zarr, such as:
            - `compressor`: A Zarr compressor instance for compressing the data.
            - `chunks`: A tuple specifying the chunk shape for writing data.
            To specify separately for each resolution level,
            pass a list of dicts with storage options for each level, e.g.
            `[{'compressor': Blosc(), 'chunks': (64, 64, 64)}, {'compressor': Blosc(), 'chunks': (128, 128, 128)}, ...]`
        version : str, optional
            The OME-Zarr format version to use. Defaults to "0.5".
        compute : bool, optional
            If True, compute immediately; otherwise return delayed objects.
        overwrite : bool, optional
            If True (default), delete and recreate the store from scratch.
            If False, skip writing data that already exists:
            - Main image data is only written if the store doesn't exist
            - Each label is only written if it doesn't already exist
            This allows progressively adding new labels to an existing image.

        Returns
        -------
        list
            If `compute` is False, returns a list of Dask delayed objects
            representing the write operations.
        """
        import os
        import shutil

        from ome_zarr.format import Format, FormatV04, FormatV05
        from ome_zarr.utils import _recursive_pop_nones
        from ome_zarr.writer import _write_pyramid_to_zarr, check_group_fmt

        delayed = []

        # Determine if store already exists
        if isinstance(group, str):
            store_exists = os.path.exists(group)
        else:
            store_exists = True  # zarr.Group was passed in, so it exists

        # Decide whether to write main image data
        write_image_data = not store_exists or overwrite

        if write_image_data:
            # Delete existing store if overwriting
            if overwrite and isinstance(group, str) and os.path.exists(group):
                shutil.rmtree(group)

            fmt: Format | None = None
            if version == "0.5" or version == "0.6":
                fmt = FormatV05()
            elif version == "0.4":
                fmt = FormatV04()
            else:
                raise ValueError(f"Unsupported OME-Zarr version: {version}")

            group, fmt = check_group_fmt(group, fmt)

            # Coerce data to dask arrays for writing
            pyramid = [
                img.data if isinstance(img.data, da.Array) else da.from_array(img.data)
                for img in self.images
            ]

            default_cs = [
                next(cs for cs in self.metadata.coordinateSystems if cs.name == self.default_coordinate_system_name)
                ][0]

            # write the actual image to disk
            delayed = _write_pyramid_to_zarr(
                pyramid=pyramid,
                group=group,
                fmt=fmt,
                storage_options=storage_options,
                axes=tuple([ax.name for ax in default_cs.axes]),
                scale=cast(dict[str, float], self.images[0].scale),
                compute=compute,
                name=self.name,
            )
        # Open existing store for updating labels only
        elif isinstance(group, str):
            group = zarr.open(group, mode="r+")

        # write labels data if passed
        if self.labels is not None:
            labels_dict = cast(dict[str, NgffMultiscales], self.labels)
            label_group = group.require_group("labels")
            for label_name, ms_labels in labels_dict.items():

                # coerce image name to name in labels dict
                ms_labels.name = label_name

                if label_name in label_group and not overwrite:
                    warnings.warn(
                        f"Label group {label_name} already exists in store. "
                        f"Skipping writing this label since overwrite=False."
                    )
                    continue

                label_subgroup = label_group.require_group(label_name)

                # Write this label (always overwrite=True here since we've
                # already decided whether to skip based on the parent's flag)
                delayed += ms_labels.to_ome_zarr(
                    group=label_subgroup,
                    storage_options=storage_options,
                    version=version,
                    compute=compute,
                    overwrite=True,
                )

        list_of_labels = (
            [str(label.name) for label in labels_dict.values()] if self.labels else []
        )

        # write the metadata to disk
        if isinstance(group, str):
            group = zarr.open(group, mode="r+")

        # Only write full metadata if we wrote image data, otherwise just update labels
        if write_image_data:
            # Create a copy of metadata with normalized paths (s0, s1, etc.)
            # to match the paths used by _write_pyramid_to_zarr
            write_datasets = tuple(
                ds.model_copy(update={"path": f"s{idx}"})
                for idx, ds in enumerate(self.metadata.datasets)
            )
            write_metadata = self.metadata.model_copy(
                update={"datasets": write_datasets}
            )

            if version == "0.4":
                # in v0.4, metadata is stored under "multiscales" attribute
                metadata_dict = write_metadata.to_version("0.4").model_dump(
                    by_alias=True
                )
                metadata_dict = _recursive_pop_nones(metadata_dict)
                metadata_dict["version"] = version
                group.attrs["multiscales"] = [metadata_dict]

                if self.omero and isinstance(self.omero, Omero):
                    omero_dict = self.omero.model_dump(by_alias=True)
                    omero_dict["version"] = version
                    group.attrs["omero"] = omero_dict
                if self.image_label and isinstance(self.image_label, Label):
                    image_label_dict = self.image_label.model_dump(by_alias=True)
                    image_label_dict["version"] = version
                    group.attrs["image-label"] = image_label_dict
                if list_of_labels:
                    group_labels = group["labels"]
                    group_labels.attrs["labels"] = list_of_labels

            elif version == "0.5":
                metadata_dict = {
                    "version": version,
                    "multiscales": [
                        _recursive_pop_nones(write_metadata.to_version("0.5").model_dump(by_alias=True))
                    ],
                }

                if self.omero and isinstance(self.omero, Omero):
                    omero_dict = self.omero.model_dump(by_alias=True)
                    omero_dict["version"] = version
                    metadata_dict["omero"] = omero_dict
                if self.image_label and isinstance(self.image_label, Label):
                    image_label_dict = self.image_label.model_dump(by_alias=True)
                    image_label_dict["version"] = version
                    metadata_dict["image-label"] = image_label_dict

                if list_of_labels:
                    group_labels = group["labels"]
                    group_labels.attrs["ome"] = {
                        "version": version,
                        "labels": list_of_labels,
                    }
                group.attrs["ome"] = metadata_dict

            elif version == "0.6":
                metadata_dict = {
                    "version": version,
                    "multiscales": [
                        _recursive_pop_nones(write_metadata.model_dump(by_alias=True))
                    ],
                }
                if self.omero and isinstance(self.omero, Omero):
                    omero_dict = self.omero.model_dump(by_alias=True)
                    omero_dict["version"] = version
                    metadata_dict["omero"] = omero_dict

                if self.image_label and isinstance(self.image_label, Label):
                    image_label_dict = self.image_label.model_dump(by_alias=True)
                    image_label_dict["version"] = version
                    metadata_dict["image-label"] = image_label_dict

                if list_of_labels:
                    metadata_dict["labels"] = list_of_labels
                group.attrs["ome"] = metadata_dict

        # Update mode: only update the labels list in metadata
        elif list_of_labels:
            group_labels = group["labels"]

            if version == "0.4":
                existing_labels = group_labels.attrs.get("labels", [])
                # Merge: add new labels not already in existing list
                merged_labels = list(existing_labels)
                for label in list_of_labels:
                    if label not in merged_labels:
                        merged_labels.append(label)
                group_labels.attrs["labels"] = merged_labels
            elif version == "0.5":
                existing_ome = group_labels.attrs.get("ome", {})
                existing_labels = existing_ome.get("labels", [])
                # Merge: add new labels not already in existing list
                merged_labels = list(existing_labels)
                for label in list_of_labels:
                    if label not in merged_labels:
                        merged_labels.append(label)
                group_labels.attrs["ome"] = {
                    "version": version,
                    "labels": merged_labels,
                }
            elif version == "0.6":
                group_labels = group["labels"]
                group_labels.attrs["ome"] = {
                    "version": version,
                    "labels": list_of_labels,
                }

        return delayed

    @classmethod
    def from_ome_zarr(
        cls,
        group: zarr.Group | str,
    ) -> NgffMultiscales:
        """
        Load a multiscale pyramid from an OME-Zarr group.

        Parameters
        ----------
        group : zarr.Group or str
            The Zarr group or path containing the OME-Zarr data.

        Returns
        -------
        NgffMultiscales`
            A :class:`NgffMultiscales` container with the loaded images and metadata.
        """
        from ome_zarr.utils import _get_version

        if isinstance(group, str):
            group = zarr.open(group, mode="r")

        version = _get_version(group)

        list_of_labels = []
        omero_dict = None
        image_label_dict = None

        # This is purely for backwards compatibility
        # need to handle loading older metadata explicitly here
        if version in ("0.1", "0.2", "0.3"):
            metadata = cls._read_legacy_metadata(group, version)

            if "labels" in group:
                labels_json = group["labels"].attrs.get("labels", [])
                list_of_labels = labels_json if isinstance(labels_json, list) else []
            if "omero" in group.attrs:
                omero_dict = group.attrs.get("omero", None)
            if "image-label" in group.attrs:
                image_label_dict = group.attrs.get("image-label", None)

        elif version == "0.4":
            from ome_zarr_models.v04.multiscales import Multiscale as Multiscalev04

            metadata_json = cast(dict, group.attrs.get("multiscales", [None])[0])
            metadata = Multiscalev04.model_validate(metadata_json).to_version("0.5")

            if "labels" in group:
                labels_json = group["labels"].attrs.get("labels", [])
                list_of_labels = labels_json if isinstance(labels_json, list) else []
            if "omero" in group.attrs:
                omero_dict = group.attrs.get("omero", None)
            if "image-label" in group.attrs:
                image_label_dict = group.attrs.get("image-label", None)

        elif version == "0.5":
            from ome_zarr_models.v05.multiscales import Multiscale as Multiscalev05

            ome_attrs = cast(dict, group.attrs.get("ome", {}))
            metadata_json = ome_attrs.get("multiscales", [None])[0]
            metadata = Multiscalev05.model_validate(metadata_json)

            if "labels" in group:
                labels_ome_attrs = group["labels"].attrs.get("ome", {})
                list_of_labels = labels_ome_attrs.get("labels", [])
            if "omero" in ome_attrs:
                omero_dict = ome_attrs.get("omero", None)
            if "image-label" in ome_attrs:
                image_label_dict = ome_attrs.get("image-label", None)

        elif version == "0.6":
            ome_attrs = cast(dict, group.attrs.get("ome", {}))
            metadata_json = ome_attrs.get("multiscales", [None])[0]
            metadata = MultiscaleV06.model_validate(metadata_json)

            if "labels" in group:
                labels_json = group["labels"].attrs.get("labels", [])
                list_of_labels = labels_json if isinstance(labels_json, list) else []
            if "omero" in group.attrs:
                omero_dict = group.attrs.get("omero", None)
            if "image-label" in group.attrs:
                image_label_dict = group.attrs.get("image-label", None)

        else:
            raise ValueError(f"Unsupported OME-Zarr version: {version}")

        # get NgffImage class instances from datasets
        images = []
        for ds in metadata.datasets:
            data = da.from_zarr(group[ds.path])
            transform = ds.coordinateTransformations[0]
            
            cs = [
                next(cs for cs in metadata.coordinateSystems if cs.name == transform.output.name)
                ][0]
            
            if isinstance(transform, Scale):
                scale = transform.scale
            elif isinstance(transform, Identity):
                scale = tuple(1.0 for _ in cs.axes)
            elif isinstance(transform, TransformSequence):
                scale = transform.transformations[0].scale
            # Filter out axes with no unit, and set to None if empty
            axes_units: dict[str, str] | None = {
                ax.name: ax.unit for ax in cs.axes if ax.unit is not None
            }
            if not axes_units:
                axes_units = None
            images.append(
                NgffImage(
                    data=data,
                    axes=[ax.name for ax in cs.axes],
                    scale={d.name: s for d, s in zip(cs.axes, scale)},
                    axes_units=axes_units,
                    name=str(metadata.name),
                )
            )

        # instantiate the NgffMultiscales object without calling __post_init__
        instance = cls.__new__(cls)
        instance.images = images
        instance.metadata = metadata
        instance.omero = None
        instance.image_label = None
        instance.labels = None
        instance.method = None

        # Finally, parse omero metadata
        try:
            if omero_dict is not None:
                instance.omero = Omero.model_validate(omero_dict)
        except ValidationError as e:
            warnings.warn(f"Invalid Omero metadata: {e}")

        # parse image label metadata
        try:
            if image_label_dict is not None:
                instance.image_label = Label.model_validate(image_label_dict)
        except ValidationError as e:
            warnings.warn(f"Invalid image-label metadata: {e}")

        if metadata.name is not None:
            instance.name = metadata.name
        else:
            instance.name = "image"

        # add labels if they exist
        if list_of_labels:
            labels = {}
            for label_name in list_of_labels:
                label_group = group[f"labels/{label_name}"]
                label_multiscale = cls.from_ome_zarr(label_group)
                labels[label_name] = label_multiscale
            instance.labels = labels

        return instance

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Override setattr to ensure labels are always stored in dict format.
        """
        if name == "labels":
            value = self._parse_labels(value)
        elif name == "omero":
            value = self._parse_omero(value)
        elif name == "image_label":
            value = self._parse_image_label(value)
        super().__setattr__(name, value)

    @staticmethod
    def _parse_image_label(image_label: dict[str, Any] | Label | None) -> Label | None:
        """
        Helper method to coerce the `image_label` attribute to an instance
        of the Label model for easier processing in other methods,
        regardless of how it was originally passed by the user.
        """
        if image_label is None:
            return None
        elif isinstance(image_label, Label):
            return image_label
        elif isinstance(image_label, dict):
            try:
                # We don't want to fail the entire initialization
                # if the image label metadata is invalid, so we
                # escape possible validation errors and just warn
                # the user that the image label metadata is invalid
                return Label.model_validate(image_label)
            except ValidationError as e:
                warnings.warn(f"Invalid image-label metadata: {e}")
                return None
        else:
            raise ValueError(
                f"Invalid type for image_label: {type(image_label)}. "
                "Expected dict or Label instance."
            )

    @staticmethod
    def _parse_omero(omero: dict[str, Any] | Omero | None) -> Omero | None:
        """
        Helper method to coerce the `omero` attribute to an instance
        of the Omero model for easier processing in other methods,
        regardless of how it was originally passed by the user.
        """
        if omero is None:
            return None
        elif isinstance(omero, Omero):
            return omero
        elif isinstance(omero, dict):
            try:
                # We don't want to fail the entire initialization
                # if the omero metadata is invalid, so we
                # escape possible validation errors and just warn
                # the user that the omero metadata is invalid
                return Omero.model_validate(omero)
            except ValidationError as e:
                warnings.warn(f"Invalid Omero metadata: {e}")
                return None
        else:
            raise ValueError(
                f"Invalid type for omero: {type(omero)}. "
                "Expected dict or Omero instance."
            )

    @staticmethod
    def _parse_labels(
        labels: (
            NgffMultiscales | list[NgffMultiscales] | dict[str, NgffMultiscales] | None
        ),
    ) -> dict[str, NgffMultiscales] | None:
        """
        Helper method to coerce the `labels` attribute
        to a consistent dict format for easier processing in other methods.,
        regardless of how it was originally passed by the user.
        """

        if labels is None:
            return None
        elif isinstance(labels, NgffMultiscales):
            return {str(labels.name): labels}
        elif isinstance(labels, list):
            return {str(label.name): label for label in labels}
        elif isinstance(labels, dict):
            return labels
        else:
            raise ValueError(
                f"Invalid type for labels: {type(labels)}. "
                "Expected NgffMultiscales, list of NgffMultiscales, "
                "or dict of str to NgffMultiscales."
            )

    @staticmethod
    def _read_legacy_metadata(group, version: str) -> MultiscaleV05:
        from ome_zarr_models.v05.axes import Axis as AxisV05
        from ome_zarr_models.v05.coordinate_transformations import (
            VectorScale,
            VectorTranslation,
        )

        metadata_json = cast(dict[str, Any], group.attrs.get("multiscales", [None])[0])

        axes_map = {
            "t": AxisV05(name="t", type="time"),
            "c": AxisV05(name="c", type="channel"),
            "z": AxisV05(name="z", type="space"),
            "y": AxisV05(name="y", type="space"),
            "x": AxisV05(name="x", type="space"),
        }

        axes_order: list[str] = ["t", "c", "z", "y", "x"]
        if version == "0.3":
            axes_order_value = metadata_json.get("axes")
            if axes_order_value is None:
                raise ValueError(
                    "Metadata version 0.3 requires 'axes' field in metadata"
                )
            axes_order = cast(list[str], axes_order_value)

        axes = [axes_map[ax] for ax in axes_order]

        datasets = []
        for idx, ds in enumerate(metadata_json.get("datasets", [])):
            scale_level = [
                2.0 ** idx if s.name in ("z", "y", "x") else 1.0 for s in axes
            ]

            if idx == 0:
                transforms: tuple[VectorScale | VectorTranslation, ...] = (
                    VectorScale(type="scale", scale=scale_level),
                )
            else:
                translate = [
                    2.0 ** (idx - 1) - 0.5 if s.name in ("z", "y", "x") else 0.0
                    for s in axes
                ]
                transforms = (
                    VectorScale(type="scale", scale=scale_level),
                    VectorTranslation(type="translation", translation=translate),
                )

            datasets.append(
                Dataset(
                    path=ds.get("path", f"s{idx}"),
                    coordinateTransformations=transforms,
                )
            )

        metadata = MultiscaleV05(
            axes=axes,
            datasets=tuple(datasets),
            type=metadata_json.get("type", None),
            metadata=metadata_json.get("metadata", None),
            coordinateTransformations=None,
            name=metadata_json.get("name", "image"),
        )

        return metadata
