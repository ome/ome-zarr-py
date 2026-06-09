from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import dask.array as da
import numpy as np
import zarr
from ome_zarr_models.common.image_label_types import LabelBase as Label
from ome_zarr_models.common.omero import Omero
from ome_zarr_models.v05.multiscales import (
    Multiscale as MultiscaleV05,
)
from ome_zarr_models.v06.coordinate_transforms import (
    AnyTransform,
    Axis,
    CoordinateSystem,
    CoordinateSystemIdentifier,
    Identity,
    Scale,
    Translation,
)
from ome_zarr_models.v06.coordinate_transforms import (
    Sequence as TransformSequence,
)
from ome_zarr_models.v06.multiscales import (
    Dataset,
)
from ome_zarr_models.v06.multiscales import (
    Multiscale as MultiscaleV06,
)
from pydantic import TypeAdapter, ValidationError

from ome_zarr.scale import Methods

SPATIAL_DIMS = ["z", "y", "x"]
DEFAULT_COLORS = [
    "00FFFF",  # cyan
    "FF00FF",  # magenta
    "FFFF00",  # yellow
    "FF0000",  # red
    "00FF00",  # green
    "0000FF",  # blue
    "FFFFFF",  # white
    "FFA500",  # orange
    "800080",  # purple
    "008000",  # dark green
]


@dataclass
class OMEZarrImage:
    """
    Single-scale image representation with metadata.

    This class serves as the entrypoint to creating ome-zarr
    images on disk. The :py:class:`OMEZarrMultiscale` class and
    :py:class:`OMEZarrLabels` multi-resolution representations of
    ome-zarr images can be created from instances of this class.

    Parameters
    ----------
    data : dask.array.Array or numpy.ndarray
        The image data array. Can be a NumPy array or a Dask array.
        If a NumPy array is provided, it will be converted to a
        Dask array internally.
    axes : Sequence[str] or str
        The axis names corresponding to the data array axes,
        i.e. ('c', 'z', 'y', 'x').
    scale : dict[str, float] | None
        The physical scale for each axis, with keys as axis names,
        e.g. {'x': 0.1, 'y': 0.1, 'z': 0.5}. Missing axes are auto-set to 1.0
        with a warning. Default is None, which sets all scales to 1.0.
    axes_units : dict[str, str] | None
        Units for each axis, e.g. {'x': 'micrometer', 'y': 'micrometer'}.
        Default is None (no units).
    name : str
        Name of the image. Default is "image".

    Example
    -------
    .. code-block:: python

        import numpy as np
        data = np.random.poisson(lam=10, size=(2, 10, 128, 128)).astype(np.uint8)
        image = OMEZarrImage(
            data=data,
            axes="czyx",
            scale={"c": 1.0, "z": 0.5, "y": 0.1, "x": 0.1},
            axes_units={"c": None, "z": "micrometer", "y": "micrometer", "x": "micrometer"},
            name="my_image",
        )
    """

    data: da.Array | np.ndarray
    axes: Sequence[str] | str
    scale: dict[str, float] | None = None
    axes_units: dict[str, str] | None = None
    name: str = "image"

    def __post_init__(self):
        # coerce axes to list
        if isinstance(self.axes, str):
            self.axes = list(self.axes)

        # validate dimensions match data shape
        if len(self.axes) != len(self.data.shape):
            raise ValueError(
                f"Number of dimensions in data ({len(self.data.shape)}) "
                f"does not match number of dims ({len(self.axes)})"
            )

        # set default scale if unset
        if self.scale is None:
            self.scale = dict.fromkeys(self.axes, 1.0)

        # validate and normalize scale dict
        if (scale_set := set(self.scale)) != (axes_set := set(self.axes)):
            if diff := scale_set.difference(axes_set):
                raise ValueError(
                    f"Scale contains invalid ax(i)(e)s: {diff}. Valid axes are: {axes_set}"
                )

            warnings.warn(
                f"Scale value not provided for ax(i)(e)s '{axes_set.difference(scale_set)}'. "
                f"Using default scale of 1.0.",
                stacklevel=2,
            )

        # rebuild scale dict with defaults for missing axes
        self.scale = {d: self.scale.get(d, 1.0) for d in self.axes}

        # coerce data to dask array
        if not isinstance(self.data, da.Array):
            self.data = da.from_array(self.data)


class OMEZarrMultiscaleBase:

    name: str

    def __init__(
        self,
        image: OMEZarrImage,
        scale_factors: list[int] | tuple[int, ...] | list[dict[str, int]] | None = None,
        coordinateTransformations: (
            tuple[AnyTransform, ...] | list[dict[str, Any]] | None
        ) = None,
        method: str | Methods | None = Methods.RESIZE,
        default_coordinate_system_name: str = "physical",
    ):
        from ome_zarr.scale import _build_pyramid

        if scale_factors is None:
            scale_factors = (2, 4, 8, 16)

        self.name = image.name

        if isinstance(method, Methods):
            method = str(method.value)
        elif method is None:
            method = str(Methods.RESIZE.value)

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
                OMEZarrImage(
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
                            input=CoordinateSystemIdentifier(path=f"s{idx}"),
                            output=CoordinateSystemIdentifier(
                                name=default_coordinate_system_name
                            ),
                        ),
                    ),
                )
            )

        self._images = images

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
            name=default_coordinate_system_name,
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
                    tf.input.name != default_coordinate_system_name
                    and tf.output.name != default_coordinate_system_name
                ):
                    raise ValueError(
                        f"Coordinate transformation at index {idx} must have either "
                        f"input or output coordinate system name matching the default "
                        f"coordinate system name '{default_coordinate_system_name}': {tf}"
                    )

        self.metadata = MultiscaleV06(
            coordinateSystems=tuple([coordinate_system]),
            datasets=tuple(datasets),
            name=image.name,
            coordinateTransformations=transforms,
        )

    def to_ome_zarr(
        self,
        group: zarr.Group | str,
        storage_options: list[dict[str, Any]] | dict[str, Any] | None = None,
        version: Literal["0.6.dev4", "0.5", "0.4"] = "0.6.dev4",
        compute: bool = True,
        overwrite: bool = False,
    ) -> list:

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

            default_cs = self.metadata.intrinsic_coordinate_system

            # write the actual image to disk
            delayed += _write_pyramid_to_zarr(
                pyramid=pyramid,
                group=group,
                fmt=fmt,
                storage_options=storage_options,
                axes=tuple([ax.name for ax in default_cs.axes]),
                scale=cast(dict[str, float], self.images[0].scale),
                compute=compute,
                name=self.name,
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

            elif version == "0.5":
                metadata_dict = {
                    "version": version,
                    "multiscales": [
                        _recursive_pop_nones(
                            write_metadata.to_version("0.5").model_dump(by_alias=True)
                        )
                    ],
                }

                group.attrs["ome"] = metadata_dict

            elif version == "0.6.dev4":
                metadata_dict = {
                    "version": version,
                    "multiscales": [
                        _recursive_pop_nones(write_metadata.model_dump(by_alias=True))
                    ],
                }

        # # Update mode: only update the labels list in metadata
        # elif list_of_labels:
        #     group_labels = group["labels"]

        #     if version == "0.4":
        #         existing_labels = group_labels.attrs.get("labels", [])
        #         # Merge: add new labels not already in existing list
        #         merged_labels = list(existing_labels)
        #         for label in list_of_labels:
        #             if label not in merged_labels:
        #                 merged_labels.append(label)
        #         group_labels.attrs["labels"] = merged_labels
        #     elif version == "0.5":
        #         existing_ome = group_labels.attrs.get("ome", {})
        #         existing_labels = existing_ome.get("labels", [])
        #         # Merge: add new labels not already in existing list
        #         merged_labels = list(existing_labels)
        #         for label in list_of_labels:
        #             if label not in merged_labels:
        #                 merged_labels.append(label)
        #         group_labels.attrs["ome"] = {
        #             "version": version,
        #             "labels": merged_labels,
        #         }
        #     elif version == "0.6":
        #         group_labels = group["labels"]
        #         group_labels.attrs["ome"] = {
        #             "version": version,
        #             "labels": list_of_labels,
        #         }
        delayed += self._write_additional_meta_data(
            group=group,
            version=version,
            storage_options=storage_options,
            compute=compute,
            overwrite=overwrite,
        )

        return delayed

    @classmethod
    def from_ome_zarr(
        cls,
        group: zarr.Group | str,
    ) -> OMEZarrMultiscale | OMEZarrLabels:
        """
        Load a multiscale pyramid from an OME-Zarr group.

        Creates an instance with base attributes set, then calls
        `_read_additional_metadata` to handle class-specific metadata
        (e.g., omero for images, image-label for labels).

        Parameters
        ----------
        group : zarr.Group or str
            The Zarr group or path containing the OME-Zarr data.

        Returns
        -------
        OMEZarrMultiscale | OMEZarrLabels
            A container with the loaded images and metadata.
        """
        from ome_zarr.utils import _get_version

        if isinstance(group, str):
            opened = zarr.open(group, mode="r")
            if not isinstance(opened, zarr.Group):
                raise ValueError(f"Expected a zarr.Group but got {type(opened)}")
            group = opened

        version = _get_version(group)

        is_label = False

        # Handle loading based on version
        if version in ("0.1", "0.2", "0.3"):
            metadata = cls._read_legacy_metadata(group, version)
            if "image-label" in group.attrs:
                is_label = True

        elif version == "0.4":
            from ome_zarr_models.v04.multiscales import Multiscale as Multiscalev04

            metadata_json = cast(dict, group.attrs.get("multiscales", [None])[0])
            if metadata_json is None:
                raise ValueError(
                    "Multiscales metadata not found in group attributes. "
                    "Opening groups other than multiscales (i.e., HCS, Plates, Wells) "
                    "is currently not supported."
                )

            metadata = Multiscalev04.model_validate(metadata_json).to_version("0.6")

            if "image-label" in group.attrs:
                is_label = True

        elif version == "0.5":
            from ome_zarr_models.v05.multiscales import Multiscale as Multiscalev05

            ome_attrs = cast(dict[str, Any], group.attrs.get("ome", {}))
            metadata_json = ome_attrs.get("multiscales", [None])[0]

            if metadata_json is None:
                raise ValueError(
                    "Multiscales metadata not found in group attributes. "
                    "Opening groups other than multiscales (i.e., HCS, Plates, Wells) "
                    "is currently not supported."
                )

            metadata = Multiscalev05.model_validate(metadata_json).to_version("0.6")

            if "image-label" in ome_attrs:
                is_label = True

        elif "0.6" in version:
            from ome_zarr_models.v06.multiscales import Multiscale as Multiscalev06

            ome_attrs = cast(dict[str, Any], group.attrs.get("ome", {}))
            metadata_json = ome_attrs.get("multiscales", [None])[0]

            if metadata_json is None:
                raise ValueError(
                    "Multiscales metadata not found in group attributes. "
                    "Opening groups other than multiscales (i.e., HCS, Plates, Wells) "
                    "is currently not supported."
                )

            metadata = Multiscalev06.model_validate(metadata_json)

            if "image-label" in ome_attrs:
                is_label = True

        else:
            raise ValueError(f"Unsupported OME-Zarr version: {version}")

        # get NgffImage class instances from datasets
        images = []
        for ds in metadata.datasets:
            data = da.from_zarr(group[ds.path])
            transform = ds.coordinateTransformations[0]

            cs = metadata.intrinsic_coordinate_system

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
                OMEZarrImage(
                    data=data,
                    axes=[ax.name for ax in cs.axes],
                    scale={d.name: s for d, s in zip(cs.axes, scale)},
                    axes_units=axes_units,
                    name=str(metadata.name) if metadata.name else "image",
                )
            )

        return_cls: type[OMEZarrLabels | OMEZarrMultiscale]
        if is_label:
            return_cls = OMEZarrLabels
        else:
            return_cls = OMEZarrMultiscale

        # Create instance without calling __init__
        instance = return_cls.__new__(return_cls)
        instance._images = images
        instance.metadata = metadata
        instance.name = str(metadata.name) if metadata.name else "image"

        # Let derived classes read their specific metadata
        instance._read_additional_metadata(group, version)

        return instance

    @property
    def images(self) -> list[OMEZarrImage]:
        """
        List of images at each pyramid level.
        """
        return self._images

    def _write_additional_meta_data(
        self,
        group: zarr.Group,
        version: Literal["0.5", "0.4"] = "0.5",
        storage_options: list[dict[str, Any]] | dict[str, Any] | None = None,
        compute: bool = True,
        overwrite: bool = False,
    ) -> list:
        """
        Hook for derived classes to write additional metadata fields
        (e.g. labels, omero, image-label) to the OME-Zarr attributes after writing the main image data.

        Returns
        -------
        list
            List of delayed objects if compute=False, otherwise empty list.
        """
        return []

    @staticmethod
    def _read_legacy_metadata(group, version: str) -> MultiscaleV05:
        """Read metadata from legacy OME-Zarr versions (0.1, 0.2, 0.3)."""

        metadata_json = cast(dict[str, Any], group.attrs.get("multiscales", [None])[0])

        axes_map = {
            "t": Axis(name="t", type="time"),
            "c": Axis(name="c", type="channel"),
            "z": Axis(name="z", type="space"),
            "y": Axis(name="y", type="space"),
            "x": Axis(name="x", type="space"),
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
        cs = CoordinateSystem(axes=tuple(axes), name="physical")

        datasets = []
        for idx, ds in enumerate(metadata_json.get("datasets", [])):
            scale_level = [
                2.0 ** idx if s.name in ("z", "y", "x") else 1.0 for s in axes
            ]

            path = ds.get("path", f"s{idx}")
            if idx == 0:
                transforms: tuple[Scale | Translation, ...] = (
                    Scale(
                        type="scale",
                        scale=scale_level,
                        input=CoordinateSystemIdentifier(path=path),
                        output=CoordinateSystemIdentifier(name="physical"),
                    ),
                )
            else:
                translate = [
                    2.0 ** (idx - 1) - 0.5 if s.name in ("z", "y", "x") else 0.0
                    for s in axes
                ]
                transforms = (
                    TransformSequence(
                        transformations=(
                            Scale(type="scale", scale=scale_level),
                            Translation(type="translation", translation=translate),
                        ),
                        input=CoordinateSystemIdentifier(path=path),
                        output=CoordinateSystemIdentifier(name="physical"),
                    ),
                )

            datasets.append(
                Dataset(
                    path=path,
                    coordinateTransformations=transforms,
                )
            )

        metadata = MultiscaleV06(
            coordinateSystems=tuple(
                [
                    cs,
                ]
            ),
            datasets=tuple(datasets),
            type=metadata_json.get("type", None),
            metadata=metadata_json.get("metadata", None),
            coordinateTransformations=None,
            name=metadata_json.get("name", "image"),
        )

        return metadata

    def _read_additional_metadata(
        self,
        group: zarr.Group,
        version: str,
    ) -> None:
        """
        Hook for derived classes to read additional metadata fields
        (e.g., omero, image-label) from the OME-Zarr attributes after loading.

        Called by `from_ome_zarr` after basic loading is complete.
        """


class OMEZarrMultiscale(OMEZarrMultiscaleBase):
    """
    Container for multiscale image pyramid with OME-Zarr metadata.

    If built from an instance of :py:class:`OMEZarrImage`, the instantiation
    of this class handles the construction of the ome-zarr multi-resolution scheme
    as delayed dask arrays.
    It can be used to write such arrays and associated metadata to disk and read
    from local and remote storages.

    This class implements convenient handling of additional subgroups
    (i.e., labels) or metadata fields (i.e., the omero metadata field
    for display settings).

    Parameters
    ----------
    image : OMEZarrImage
        The OMEZarrImage instance from which to build the multi-resolution levels.
    scale_factors : list[int] | tuple[int, ...] | list[dict[str, int]] | None
        Scale factors for each pyramid level. If a list of ints or tuple is provided,
        it is applied uniformly across all spatial axes. If a list of dicts is provided,
        each dict should specify scale factors for each axis, e.g. {'x': 2, 'y': 2, 'z': 1}.
        Default is (2, 4, 8, 16).
    method : ome_zarr.scale.Methods | str | None
        Rescaling method to use when generating pyramid levels. Default is Methods.RESIZE.
    coordinateTransformations :
        Additional coordinate transformations to include in the metadata for each level.
    labels : OMEZarrLabels | list[OMEZarrLabels] | dict[str, OMEZarrLabels] | None
        Labels associated with the image. Can be a single OMEZarrLabels instance, a list of them,
        or a dict mapping label names to OMEZarrLabels instances. Default is None (no labels).
    channel_names : list[str] | None
        List of channel names corresponding to the 'c' axis, e.g. ['DAPI', 'GFP', 'RFP'].
        Default is None (no channel names).
    channel_colors : list[list[int]] | list[str] | None
        List of colors for each channel corresponding to the 'c' axis.
        Can be passed as a list of RGB values (i.e., [[255, 0, 0], [0, 255, 0], ...])
        or as hex strings (i.e., ['FF0000', '00FF00', '0000FF']).
        Default is None (no channel colors).
    contrast_limits : list[tuple[float, float]] | None
        List of contrast limits for each channel corresponding to the 'c' axis,
        e.g. [(0, 255), (0, 1000), ...].
        Default is None (no contrast limits).

    Attributes
    ----------
    images : list[OMEZarrImage]
        List of images at each pyramid level.
    labels : dict[str, OMEZarrLabels] | None
        Dictionary mapping label names to OMEZarrLabels instances, or None if no labels are associated.

    Methods
    -------
        to_ome_zarr(group, storage_options, version, compute, overwrite)
            Write the multiscale image pyramid and metadata to an OME-Zarr group.
        from_ome_zarr(group)
            Load a multiscale image pyramid and metadata from an OME-Zarr group.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from ome_zarr import OMEZarrImage, OMEZarrMultiscale
        data = np.random.poisson(lam=10, size=(2, 10, 128, 128)).astype(np.uint8)
        image = OMEZarrImage(
            data=data,
            axes="czyx",
        )
        multiscale = OMEZarrMultiscale(
            image=image,
            scale_factors=[2, 4, 8, 16],
            channel_names=["DAPI", "GFP"]
        )
    """

    def __init__(
        self,
        image: OMEZarrImage,
        scale_factors: list[int] | tuple[int, ...] | list[dict[str, int]] | None = None,
        method: str | Methods | None = Methods.RESIZE,
        coordinateTransformations: list[Scale | Translation | Identity] | None = None,
        labels: (
            OMEZarrLabels | list[OMEZarrLabels] | dict[str, OMEZarrLabels] | None
        ) = None,
        channel_names: list[str] | None = None,
        channel_colors: list[list[int]] | list[str] | None = None,
        contrast_limits: list[tuple[float, float]] | None = None,
    ):
        super().__init__(
            image=image,
            scale_factors=scale_factors,
            method=method,
            coordinateTransformations=coordinateTransformations,
        )

        # Normalize labels to dict format
        self._labels = self._parse_labels(labels)

        # Parse omero metadata from channel parameters
        self._omero = None
        self._parse_omero_metadata(channel_names, channel_colors, contrast_limits)

    def _write_additional_meta_data(
        self,
        group: zarr.Group,
        version: Literal["0.5", "0.4"] = "0.5",
        storage_options: list[dict[str, Any]] | dict[str, Any] | None = None,
        compute: bool = True,
        overwrite: bool = False,
    ) -> list:
        from ome_zarr.utils import _recursive_pop_nones

        delayed: list = []

        # Write omero metadata
        if self._omero and isinstance(self._omero, Omero):
            omero_dict = _recursive_pop_nones(self._omero.model_dump(by_alias=True))

            if version == "0.4":
                group.attrs["omero"] = omero_dict
            elif version == "0.5":
                if "ome" not in group.attrs:
                    raise ValueError("OME-Zarr attributes not found in group")
                ome = cast(dict, group.attrs["ome"])
                omero_dict["version"] = version
                ome["omero"] = omero_dict
                group.attrs["ome"] = ome

        # Write labels if present
        if self._labels is not None:
            label_group = group.require_group("labels")
            list_of_labels: list[str] = []

            for label_name, ms_labels in self._labels.items():
                # Coerce image name to match label name in dict
                ms_labels.name = label_name
                list_of_labels.append(label_name)

                # Skip if label already exists and overwrite=False
                if label_name in label_group and not overwrite:
                    warnings.warn(
                        f"Label group {label_name} already exists in store. "
                        f"Skipping writing this label since overwrite=False."
                    )
                    continue

                label_subgroup = label_group.require_group(label_name)

                # Write this label's pyramid and metadata
                # Always overwrite=True here since we've already decided
                # whether to skip based on the parent's flag
                delayed += ms_labels.to_ome_zarr(
                    group=label_subgroup,
                    storage_options=storage_options,
                    version=version,
                    compute=compute,
                    overwrite=True,
                )

            # Update labels list in metadata
            if version == "0.4":
                label_group.attrs["labels"] = list_of_labels
            elif version == "0.5":
                label_group.attrs["ome"] = {
                    "version": version,
                    "labels": list_of_labels,
                }

        return delayed

    def _parse_omero_metadata(
        self,
        channel_names: list[str] | None,
        channel_colors: list[list[int]] | list[str] | None,
        contrast_limits: list[tuple[float, float]] | None,
    ) -> None:
        """
        Build omero metadata from channel parameters.
        """
        if "c" not in self._images[0].axes:
            n_channels = 1
        else:
            # Make default values and then replace with provided values
            channel_axis = self._images[0].axes.index("c")
            n_channels = self._images[0].data.shape[channel_axis]

        # Make sure that all channel descriptors line up with the data dimensions
        for param in [channel_names, channel_colors, contrast_limits]:
            if param is not None and len(param) != n_channels:
                raise ValueError(
                    f"Length of {param} ({len(param)}) does not match "
                    f"number of channels ({n_channels})"
                )

        channel_metadata = []
        for i in range(n_channels):
            if channel_names is not None:
                name = channel_names[i]
            else:
                name = f"Channel {i}"

            if channel_colors is not None:
                color = channel_colors[i]
                # Coerce RGBA/RGB list values to hex strings
                if isinstance(color, (list, tuple)):
                    # Convert RGB/RGBA to hex, taking first
                    # 3 values and ignoring alpha
                    color = f"{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            else:
                color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]

            color = color.lstrip("#")  # Remove # if present

            dtype_max = self._images[0].data.dtype.itemsize * 255
            if contrast_limits is not None:
                channel_contrast = contrast_limits[i]
            else:
                channel_contrast = (
                    0,
                    dtype_max,
                )  # TODO: best way to get max value from dtype?

            channel_metadata.append(
                {
                    "label": name,
                    "active": True,
                    "color": color,
                    "window": {
                        "min": 0,
                        "start": channel_contrast[0],
                        "max": dtype_max,
                        "end": channel_contrast[1],
                    },
                }
            )

        try:
            self._omero = Omero.model_validate({"channels": channel_metadata})
        except ValidationError as e:
            warnings.warn(f"Failed to validate Omero metadata: {e}")

    @property
    def labels(self) -> dict[str, OMEZarrLabels] | None:
        return self._labels

    @labels.setter
    def labels(
        self,
        value: OMEZarrLabels | list[OMEZarrLabels] | dict[str, OMEZarrLabels] | None,
    ):
        self._labels = self._parse_labels(value)

    @property
    def omero(self) -> Omero | None:
        return self._omero

    @omero.setter
    def omero(self, value: Omero | dict[str, Any] | None):
        if isinstance(value, dict):
            self._omero = Omero.model_validate(value)
        else:
            self._omero = value

    @staticmethod
    def _parse_labels(
        labels: OMEZarrLabels | list[OMEZarrLabels] | dict[str, OMEZarrLabels] | None,
    ) -> dict[str, OMEZarrLabels] | None:
        if labels is None:
            return None
        elif isinstance(labels, OMEZarrLabels):
            return {str(labels.name): labels}
        elif isinstance(labels, list):
            return {str(label.name): label for label in labels}
        elif isinstance(labels, dict):
            return labels
        else:
            raise ValueError(
                "Invalid type for labels. Expected OMEZarrLabels, "
                "list of OMEZarrLabels, or dict of OMEZarrLabels."
            )

    def _read_additional_metadata(
        self,
        group: zarr.Group,
        version: str,
    ) -> None:
        """Read omero metadata and load labels."""
        # Initialize class-specific attributes
        self._labels = None
        self._omero = None

        # Read omero metadata
        omero_dict: dict[str, Any] | None = None

        if version in ("0.1", "0.2", "0.3", "0.4") and "omero" in group.attrs:
            omero_dict = cast(dict[str, Any] | None, group.attrs.get("omero", None))
        elif version == "0.5":
            ome_attrs = cast(dict[str, Any], group.attrs.get("ome", {}))
            if "omero" in ome_attrs:
                omero_dict = cast(dict[str, Any] | None, ome_attrs.get("omero", None))

        if omero_dict is not None:
            try:
                self._omero = Omero.model_validate(omero_dict)
            except ValidationError as e:
                warnings.warn(f"Invalid Omero metadata: {e}")

        # Read labels list
        list_of_labels: list[str] = []

        if version in ("0.1", "0.2", "0.3", "0.4") and "labels" in group:
            labels_json = group["labels"].attrs.get("labels", [])
            list_of_labels = (
                cast(list[str], labels_json) if isinstance(labels_json, list) else []
            )
        elif version == "0.5" and "labels" in group:
            labels_ome_attrs = cast(
                dict[str, Any], group["labels"].attrs.get("ome", {})
            )
            list_of_labels = cast(list[str], labels_ome_attrs.get("labels", []))

        # Load labels if they exist
        if list_of_labels:
            loaded_labels: dict[str, OMEZarrLabels] = {}
            for label_name in list_of_labels:
                label_subgroup = group[f"labels/{label_name}"]
                if not isinstance(label_subgroup, zarr.Group):
                    warnings.warn(f"Label {label_name} is not a zarr.Group, skipping")
                    continue
                label_multiscale = cast(
                    OMEZarrLabels, OMEZarrLabels.from_ome_zarr(label_subgroup)
                )
                loaded_labels[label_name] = label_multiscale
            self._labels = loaded_labels


class OMEZarrLabels(OMEZarrMultiscaleBase):
    """
    Container for label images with OME-Zarr metadata.

    This class extends OMEZarrMultiscaleBase and implements
    handling of additional metadata fields specific to label images,
    such as image-label metadata.

    Parameters
    ----------
    image : OMEZarrImage
    scale_factors : list[int] | tuple[int, ...] | list[dict[str, int]] | None, optional
        Scale factors for each pyramid level. If a list of ints or tuple is provided,
        it is applied uniformly across all spatial axes. If a list of dicts is provided,
        each dict should specify scale factors for each axis, e.g. {'x': 2, 'y': 2, 'z': 1}.
        Default is (2, 4, 8, 16).
    method : str | ome_zarr.scale.Methods, optional
        Rescaling method to use when generating pyramid levels. Default is Methods.NEAREST,
        since these are labels.
    auto_parse_labels : bool, optional
        Whether to automatically inspect the data for present label values and write these
        to the metadata. This can be time consuming for large datasets, so it is optional.
        Default is True.

     Attributes
    ----------
    images : list[OMEZarrImage]
        List of label images at each pyramid level.
    image_label : Label | None
        Optional image-label metadata for rendering label images, or None if not provided.

    """

    _image_label: Label | None

    def __init__(
        self,
        image: OMEZarrImage,
        scale_factors: list[int] | tuple[int, ...] | list[dict[str, int]] | None = None,
        method: str | Methods | None = Methods.NEAREST,
        auto_parse_labels: bool = True,
    ):
        super().__init__(
            image=image,
            scale_factors=scale_factors,
            method=method,
            coordinateTransformations=None,
        )

        # Build image-label metadata if auto_parse_labels is enabled
        self._image_label = None
        if auto_parse_labels:
            self._parse_image_label_metadata()

    def _parse_image_label_metadata(self) -> None:
        """Build image-label metadata by inspecting unique label values."""
        label_values = da.unique(self._images[0].data).compute().tolist()
        colors = [
            {
                "label-value": label,
                "rgba": [np.random.randint(0, 255) for _ in range(3)] + [255],
            }
            for label in label_values
        ]

        self._image_label = Label.model_validate(
            {
                "colors": colors,
                "source": {"image": "../.."},
                "properties": [{"label-value": i} for i in label_values],
            }
        )

    @property
    def image_label(self) -> Label | None:
        return self._image_label

    @image_label.setter
    def image_label(self, value: Label | dict[str, Any] | None):
        if isinstance(value, dict):
            self._image_label = Label.model_validate(value)
        else:
            self._image_label = value

    def _write_additional_meta_data(
        self,
        group: zarr.Group,
        version: Literal["0.5", "0.4"] = "0.5",
        storage_options: list[dict[str, Any]] | dict[str, Any] | None = None,
        compute: bool = True,
        overwrite: bool = False,
    ) -> list:
        from ome_zarr.utils import _recursive_pop_nones

        if self._image_label is not None and isinstance(self._image_label, Label):
            if version == "0.4":
                group.attrs["image-label"] = _recursive_pop_nones(
                    self._image_label.model_dump(by_alias=True)
                )
            elif version == "0.5":
                ome = cast(dict, group.attrs.get("ome", {}))
                ome["image-label"] = _recursive_pop_nones(
                    self._image_label.model_dump(by_alias=True)
                )
                group.attrs["ome"] = ome

        return []

    def _read_additional_metadata(
        self,
        group: zarr.Group,
        version: str,
    ) -> None:
        """Read image-label metadata."""
        # Initialize class-specific attributes
        self._image_label = None

        image_label_dict: dict[str, Any] | None = None

        if version in ("0.1", "0.2", "0.3", "0.4"):
            if "image-label" in group.attrs:
                image_label_dict = cast(
                    dict[str, Any] | None, group.attrs.get("image-label", None)
                )
        elif version == "0.5":
            ome_attrs = cast(dict[str, Any], group.attrs.get("ome", {}))
            if "image-label" in ome_attrs:
                image_label_dict = cast(
                    dict[str, Any] | None, ome_attrs.get("image-label", None)
                )

        if image_label_dict is not None:
            try:
                self._image_label = Label.model_validate(image_label_dict)
            except ValidationError as e:
                warnings.warn(f"Invalid image-label metadata: {e}")
