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
from ome_zarr_models.v05.axes import (
    Axis,
)
from ome_zarr_models.v05.coordinate_transformations import (
    Identity as Identity,
)
from ome_zarr_models.v05.coordinate_transformations import (
    VectorScale as Scale,
)
from ome_zarr_models.v05.coordinate_transformations import (
    VectorTranslation as Translation,
)
from ome_zarr_models.v05.multiscales import (
    Dataset,
)
from ome_zarr_models.v05.multiscales import (
    Multiscale as MultiscaleV05,
)
from pydantic import ValidationError

from ome_zarr.scale import Methods

SPATIAL_DIMS = ["z", "y", "x"]
DEFAULT_COLORS = [
    "#00FFFF",  # cyan
    "#FF00FF",  # magenta
    "#FFFF00",  # yellow
    "#FF0000",  # red
    "#00FF00",  # green
    "#0000FF",  # blue
    "#FFFFFF",  # white
    "#FFA500",  # orange
    "#800080",  # purple
    "#008000",  # dark green
]


@dataclass
class OMEZarrImage:
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
    channel_names : list of str, optional
        List of channel names corresponding to the 'c' axis, e.g. ['DAPI', 'GFP', 'RFP'].
        Default is None (no channel names).
        Only relevant for intensity images (i.e., no label images.)
    channel_colors : list of list of int or list of str, optional
        List of colors for each channel corresponding to the 'c' axis.
        Can be passed as a list of RGB values (i.e., [[255, 0, 0], [0, 255, 0], ...])
        or as hex strings (i.e., ['#FF0000', '#00FF00', '#0000FF']).
        Default is None (no channel colors).
        Only relevant for intensity images (i.e., no label images.)
    contrast_limits : list of tuple of float, optional
        List of contrast limits for each channel corresponding to the 'c' axis,
        e.g. [(0, 255), (0, 1000), ...].
        Default is None (no contrast limits).
        Only relevant for intensity images (i.e., no label images.)
    """

    data: da.Array | np.ndarray
    axes: Sequence[str] | str
    scale: Sequence[float] | dict[str, float] | None = None
    axes_units: dict[str, str] | None = None
    name: str = "image"
    channel_names: list[str] | None = None
    channel_colors: list[list[int]] | list[str] | None = None
    contrast_limits: list[tuple[float, float]] | None = None

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

        if self.channel_names is not None:
            if len(self.channel_names) > 0 and "c" not in self.axes:
                raise ValueError(
                    f"Channel names provided but 'c' axis not found in axes {self.axes}"
                )
        if self.channel_colors is not None:
            if len(self.channel_colors) > 0 and "c" not in self.axes:
                raise ValueError(
                    f"Channel colors provided but 'c' axis not found in axes {self.axes}"
                )
        if self.contrast_limits is not None:
            if len(self.contrast_limits) > 0 and "c" not in self.axes:
                raise ValueError(
                    f"Contrast limits provided but 'c' axis not found in axes {self.axes}"
                )


class OMEZarrMultiscaleBase:
    """
    Base class for multiscale image pyramid with OME-Zarr metadata.

    Parameters
    """

    name: str

    def __init__(
        self,
        image: OMEZarrImage,
        scale_factors: list[int] | tuple[int, ...] | list[dict[str, int]] | None = None,
        method: str | Methods | None = Methods.RESIZE,
        coordinateTransformations: list[Scale | Translation | Identity] | None = None,
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
                    channel_names=image.channel_names,
                    channel_colors=image.channel_colors,
                    contrast_limits=image.contrast_limits,
                )
            )
            datasets.append(
                Dataset(
                    path=f"s{idx}",
                    coordinateTransformations=(
                        Scale(
                            type="scale",
                            scale=list(level_scale.values()),
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

        self.metadata = MultiscaleV05(
            axes=tuple(axes),
            datasets=tuple(datasets),
            name=image.name,
            coordinateTransformations=coordinateTransformations,
        )

        self._parse_additional_metadata()

    def to_ome_zarr(
        self,
        group: zarr.Group | str,
        storage_options: list[dict[str, Any]] | dict[str, Any] | None = None,
        version: Literal["0.5", "0.4"] = "0.5",
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
            if version == "0.5":
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

            # write the actual image to disk
            delayed = _write_pyramid_to_zarr(
                pyramid=pyramid,
                group=group,
                storage_options=storage_options,
                fmt=fmt,
                scale=cast(dict[str, float], self.images[0].scale),
                axes=[dict(ax) for ax in self.metadata.axes],
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
                        _recursive_pop_nones(write_metadata.model_dump(by_alias=True))
                    ],
                }

                group.attrs["ome"] = metadata_dict

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
            metadata = Multiscalev04.model_validate(metadata_json).to_version("0.5")

            if "image-label" in group.attrs:
                is_label = True

        elif version == "0.5":
            from ome_zarr_models.v05.multiscales import Multiscale as Multiscalev05

            ome_attrs = cast(dict[str, Any], group.attrs.get("ome", {}))
            metadata_json = ome_attrs.get("multiscales", [None])[0]
            metadata = Multiscalev05.model_validate(metadata_json)

            if "image-label" in ome_attrs:
                is_label = True

        else:
            raise ValueError(f"Unsupported OME-Zarr version: {version}")

        # Create OMEZarrImage instances for each dataset
        images: list[OMEZarrImage] = []
        for dataset in metadata.datasets:
            path = dataset.path
            data = da.from_zarr(group[path])
            coord_transform = dataset.coordinateTransformations[0]
            scale = cast(list[float], coord_transform.scale)
            # Filter out axes with no unit, and set to None if empty
            axes_units: dict[str, str] | None = {
                str(ax.name): str(ax.unit)
                for ax in metadata.axes
                if ax.unit is not None and ax.name is not None
            }
            if not axes_units:
                axes_units = None
            axes_names = [str(ax.name) for ax in metadata.axes if ax.name is not None]
            images.append(
                OMEZarrImage(
                    data=data,
                    axes=axes_names,
                    scale={
                        str(ax.name): s
                        for ax, s in zip(metadata.axes, scale)
                        if ax.name is not None
                    },
                    axes_units=axes_units,
                    name=str(metadata.name) if metadata.name else "image",
                )
            )

        if is_label:
            cls = OMEZarrLabels
        else:
            cls = OMEZarrMultiscale

        # Create instance without calling __init__
        instance = cls.__new__(cls)
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

    def _parse_additional_metadata(self):
        """
        Hook for derived classes to parse additional metadata fields on initialization
        (e.g. labels, omero, image-label) from the base class after initialization.
        """

    @staticmethod
    def _read_legacy_metadata(group, version: str) -> MultiscaleV05:
        """Read metadata from legacy OME-Zarr versions (0.1, 0.2, 0.3)."""
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

    This class extends OMEZarrMultiscaleBase and implements
    handling of additional metadata fields such as labels, omero, and image-label.

    Parameters
    ----------
    image : OMEZarrImage
    scale_factors : list[int] | tuple[int, ...] | list[dict[str, int]] | None, optional
        Scale factors for each pyramid level. If a list of ints or tuple is provided,
        it is applied uniformly across all spatial axes. If a list of dicts is provided,
        each dict should specify scale factors for each axis, e.g. {'x': 2, 'y': 2, 'z': 1}.
        Default is (2, 4, 8, 16).
    method : str | Methods, optional
        Rescaling method to use when generating pyramid levels. Default is Methods.RESIZE.
    coordinateTransformations : list[Scale | Translation | Identity], optional
        Additional coordinate transformations to include in the metadata for each level.
    labels : OMEZarrLabels | list[OMEZarrLabels] | dict[str, OMEZarrLabels] | None, optional
        Labels associated with the image. Can be a single OMEZarrLabels instance, a list of them,
        or a dict mapping label names to OMEZarrLabels instances. Default is None (no labels).

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

    """

    _labels: dict[str, OMEZarrLabels] | None
    _omero: Omero | None

    def __init__(
        self,
        image: OMEZarrImage,
        scale_factors: list[int] | tuple[int, ...] | list[dict[str, int]] | None = None,
        method: str | Methods | None = Methods.RESIZE,
        coordinateTransformations: list[Scale | Translation | Identity] | None = None,
        labels: (
            OMEZarrLabels | list[OMEZarrLabels] | dict[str, OMEZarrLabels] | None
        ) = None,
    ):
        # Normalize labels to dict format
        self._labels = self._parse_labels(labels)

        super().__init__(
            image=image,
            scale_factors=scale_factors,
            method=method,
            coordinateTransformations=coordinateTransformations,
        )

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
            omero_dict["version"] = version

            if version == "0.4":
                group.attrs["omero"] = omero_dict
            elif version == "0.5":
                if "ome" not in group.attrs:
                    raise ValueError("OME-Zarr attributes not found in group")
                ome = cast(dict, group.attrs["ome"])
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

    def _parse_additional_metadata(self):
        """
        Helper function to parse metadata fields that are specific
        for images (as opposed to labels), i.e., omero metadata
        """

        # omero first
        self._omero = None
        if "c" not in self._images[0].axes:
            return

        # make sure channel_names, display_colors and contrast_limits are lists of the same length if provided
        if self._images[0].channel_names is not None:
            if self._images[0].channel_colors is not None:
                if len(self._images[0].channel_names) != len(
                    self._images[0].channel_colors
                ):
                    raise ValueError(
                        f"Length of channel_names ({len(self._images[0].channel_names)}) does not match length of channel_colors ({len(self._images[0].channel_colors)})"
                    )
            if self._images[0].contrast_limits is not None:
                if len(self._images[0].channel_names) != len(
                    self._images[0].contrast_limits
                ):
                    raise ValueError(
                        f"Length of channel_names ({len(self._images[0].channel_names)}) does not match length of contrast_limits ({len(self._images[0].contrast_limits)})"
                    )

        # make default values and then replace with provided values
        channel_axis = self._images[0].axes.index("c")
        n_channels = self._images[0].data.shape[channel_axis]

        channel_metadata = []
        for i in range(n_channels):
            if self._images[0].channel_names is not None:
                name = self._images[0].channel_names[i]
            else:
                name = f"Channel {i}"

            if self._images[0].channel_colors is not None:
                color = self._images[0].channel_colors[i]
                # Coerce RGBA/RGB list values to hex strings
                if isinstance(color, (list, tuple)):
                    # Convert RGB/RGBA to hex, taking first 3 values and ignoring alpha
                    color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            else:
                color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]

            dtype_max = self._images[0].data.dtype.itemsize * 255
            if self._images[0].contrast_limits is not None:
                contrast_limits = self._images[0].contrast_limits[i]
            else:
                contrast_limits = (
                    0,
                    dtype_max,
                )  # TODO: Check to see if this is the best way to get max value from dtype

            channel_metadata.append(
                {
                    "label": name,
                    "color": color,
                    "window": {
                        "min": 0,
                        "start": contrast_limits[0],
                        "max": dtype_max,
                        "end": contrast_limits[1],
                    },
                }
            )

        try:
            self._omero = Omero.model_validate({"channels": channel_metadata})
        except Exception as e:
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

        if version in ("0.1", "0.2", "0.3", "0.4"):
            if "omero" in group.attrs:
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

        if version in ("0.1", "0.2", "0.3", "0.4"):
            if "labels" in group:
                labels_json = group["labels"].attrs.get("labels", [])
                list_of_labels = (
                    cast(list[str], labels_json)
                    if isinstance(labels_json, list)
                    else []
                )
        elif version == "0.5":
            if "labels" in group:
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
    method : str | Methods, optional
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
        self._image_label = None
        self._auto_parse_labels = auto_parse_labels
        super().__init__(
            image=image,
            scale_factors=scale_factors,
            method=method,
            coordinateTransformations=None,
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

    def _parse_additional_metadata(self):
        if self._auto_parse_labels:
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


# @dataclass
# class OMEZarrMultiscale:
#     """
#     Container for multiscale image pyramid with OME-Zarr metadata.

#     Parameters
#     ----------
#     image : OMEZarrImage
#         The base (highest resolution) image.
#     scale_factors : list of int, optional
#         Downsampling factors for each pyramid level.
#         If passed as a list of integers (i.e. [2, 4, 8]),
#         the same factors will be applied to all *spatial* dimensions
#         except for the z-axis (if present).
#         To customize this behavior, pass a list of dicts mapping dimension names to factors, e.g.
#         `[{'x': 2, 'y': 2}, {'x': 4, 'y': 4}, {'x': 8, 'y': 8}]`
#         Default: [2, 4, 8, 16].
#     method : str or Methods, optional
#         Downsampling method to use. Default: Methods.RESIZE.
#     labels : :class:`NgffMultiscales` or dict of str to :class:`NgffMultiscales`, optional
#         Optional labels to associate with the image pyramid.
#         Can be a single :class:`NgffMultiscales` instance (for a single label pyramid)
#         or a dict mapping label names to :class:`NgffMultiscales` instances
#         (for multiple label pyramids), e.g.
#         `{'nuclei': nuclei_multiscale, 'cells': cells_multiscale}`.
#         Default is None (no labels).
#     omero : dict or Omero, optional
#         Optional Omero metadata to include in the OME-Zarr attributes.
#         Can be passed as a dict or an instance of the [Omero model](https://ome-zarr-models-py.readthedocs.io/en/stable/api/v04/image/#omero-metadata)
#         Default is None (no Omero metadata).
#         For example metadata, see [ngff specification](https://ngff.openmicroscopy.org/specifications/0.5/index.html#omero-metadata-transitional)
#     image_label : dict or Label, optional
#         Optional image-label metadata to describe rendering options specifically for label images.
#         Can signal to viewers that this image should be rendered as labels.
#         Can be passed as a dict or an instance of the [Label model](https://ome-zarr-models-py.readthedocs.io/en/stable/api/v05/image-label/)
#         For example metadata, see [ngff specification](https://ngff.openmicroscopy.org/specifications/0.5/index.html#labels-metadata)

#     Attributes
#     ----------
#     images : list of NgffImage
#         List of images at each pyramid level.
#     metadata : Multiscale
#         OME-Zarr multiscale metadata.
#     omero : Omero or None
#         Optional Omero metadata included in the OME-Zarr attributes.
#     image_label : Label or None
#         Optional image-label metadata included in the OME-Zarr attributes.
#     """

#     # Attributes that are populated in __post_init__ and not passed by the user
#     images: list[OMEZarrImage] = field(init=False)

#     # InitVars for parameters passed by the user
#     image: InitVar[OMEZarrImage]
#     scale_factors: InitVar[
#         list[int] | tuple[int, ...] | list[dict[str, int]] | None
#     ] = None
#     method: str | Methods | None = Methods.RESIZE
#     coordinateTransformations: InitVar[list[Scale | Translation | Identity] | None] = (
#         None
#     )
#     labels: (
#         OMEZarrMultiscale | list[OMEZarrMultiscale] | dict[str, OMEZarrMultiscale] | None
#     ) = None
#     omero: dict[str, Any] | Omero | None = field(init=False, default=None)
#     image_label: dict[str, Any] | Label | None = field(init=False, default=None)

#     def __post_init__(
#         self,
#         image: OMEZarrImage,
#         scale_factors: list[int] | tuple[int, ...] | list[dict[str, int]] | None,
#         coordinateTransformations: list[Scale | Translation | Identity] | None,
#     ):


#         if scale_factors is None:
#             scale_factors = (2, 4, 8, 16)

#         self.name = image.name
#         method = self.method

#         if isinstance(method, Methods):
#             method = str(method.value)
#         elif method is None:
#             method = str(Methods.RESIZE.value)

#         if self.image_label is not None and method != str(Methods.NEAREST.value):
#             warnings.warn(
#                 f"Image label metadata provided but method is {method}, not 'nearest'. "
#                 f"Consider using method='nearest' label image multiresolution pyramids."
#             )

#         # Build the pyramid data
#         pyramid = _build_pyramid(
#             image=image.data,
#             dims=image.axes,
#             scale_factors=scale_factors,
#             method=method,
#         )

#         # build scales for each level based on the original image shape
#         # and the pyramid level shapes
#         scales = []
#         # image.scale is guaranteed to be a dict after NgffImage.__post_init__
#         image_scale = image.scale
#         assert isinstance(image_scale, dict)
#         for shape in [d.shape for d in pyramid]:
#             scale = [full / level for full, level in zip(image.data.shape, shape)]
#             scales.append(
#                 {
#                     d: s * image_scale[d] if d in image_scale else 1.0
#                     for d, s in zip(image.axes, scale)
#                 }
#             )

#         # Create Image instances for each pyramid level
#         images = []
#         datasets = []
#         for idx, (level_data, level_scale) in enumerate(zip(pyramid, scales)):

#             images.append(
#                 OMEZarrImage(
#                     data=level_data,
#                     axes=image.axes,
#                     scale=level_scale,
#                     axes_units=image.axes_units,
#                     name=image.name,
#                 )
#             )
#             datasets.append(
#                 Dataset(
#                     path=f"s{idx}",
#                     coordinateTransformations=(
#                         Scale(
#                             type="scale",
#                             scale=list(level_scale.values()),
#                         ),
#                     ),
#                 )
#             )

#         self.images = images

#         # Build axes metadata
#         if image.axes_units is None:
#             image.axes_units = {}

#         axes = []
#         for d in image.axes:
#             if d in SPATIAL_DIMS:
#                 axes.append(Axis(name=d, type="space", unit=image.axes_units.get(d)))
#             elif d == "t":
#                 axes.append(Axis(name=d, type="time", unit=image.axes_units.get(d)))
#             elif d == "c":
#                 axes.append(Axis(name=d, type="channel", unit=image.axes_units.get(d)))
#             else:
#                 axes.append(Axis(name=d, type="custom", unit=image.axes_units.get(d)))

#         self.metadata = MultiscaleV05(
#             axes=tuple(axes),
#             datasets=tuple(datasets),
#             name=image.name,
#             coordinateTransformations=coordinateTransformations,
#         )

#         # coerce labels to dict if it's a single NgffMultiscales or a list
#         self.labels = self._parse_labels(self.labels)
#         self.omero = self._parse_omero(self.omero)
#         self.image_label = self._parse_image_label(self.image_label)

#     def to_ome_zarr(
#         self,
#         group: zarr.Group | str,
#         storage_options: list[dict[str, Any]] | dict[str, Any] | None = None,
#         version: str | None = "0.5",
#         compute: bool = True,
#         overwrite: bool = True,
#     ) -> list:
#         """
#         Serialize the multiscale pyramid to an OME-Zarr group.

#         Parameters
#         ----------
#         group : zarr.Group or str
#             The target Zarr group or path where the OME-Zarr data will be written.
#         storage_options : dict or list of dict, optional
#             Additional storage options to pass to Zarr, such as:
#             - `compressor`: A Zarr compressor instance for compressing the data.
#             - `chunks`: A tuple specifying the chunk shape for writing data.
#             To specify separately for each resolution level,
#             pass a list of dicts with storage options for each level, e.g.
#             `[{'compressor': Blosc(), 'chunks': (64, 64, 64)}, {'compressor': Blosc(), 'chunks': (128, 128, 128)}, ...]`
#         version : str, optional
#             The OME-Zarr format version to use. Defaults to "0.5".
#         compute : bool, optional
#             If True, compute immediately; otherwise return delayed objects.
#         overwrite : bool, optional
#             If True (default), delete and recreate the store from scratch.
#             If False, skip writing data that already exists:
#             - Main image data is only written if the store doesn't exist
#             - Each label is only written if it doesn't already exist
#             This allows progressively adding new labels to an existing image.

#         Returns
#         -------
#         list
#             If `compute` is False, returns a list of Dask delayed objects
#             representing the write operations.
#         """
#         import os
#         import shutil

#         from ome_zarr.format import Format, FormatV04, FormatV05
#         from ome_zarr.utils import _recursive_pop_nones
#         from ome_zarr.writer import _write_pyramid_to_zarr, check_group_fmt

#         delayed = []

#         # Determine if store already exists
#         if isinstance(group, str):
#             store_exists = os.path.exists(group)
#         else:
#             store_exists = True  # zarr.Group was passed in, so it exists

#         # Decide whether to write main image data
#         write_image_data = not store_exists or overwrite

#         if write_image_data:
#             # Delete existing store if overwriting
#             if overwrite and isinstance(group, str) and os.path.exists(group):
#                 shutil.rmtree(group)

#             fmt: Format | None = None
#             if version == "0.5":
#                 fmt = FormatV05()
#             elif version == "0.4":
#                 fmt = FormatV04()
#             else:
#                 raise ValueError(f"Unsupported OME-Zarr version: {version}")

#             group, fmt = check_group_fmt(group, fmt)

#             # Coerce data to dask arrays for writing
#             pyramid = [
#                 img.data if isinstance(img.data, da.Array) else da.from_array(img.data)
#                 for img in self.images
#             ]

#             # write the actual image to disk
#             delayed = _write_pyramid_to_zarr(
#                 pyramid=pyramid,
#                 group=group,
#                 storage_options=storage_options,
#                 fmt=fmt,
#                 scale=cast(dict[str, float], self.images[0].scale),
#                 axes=[dict(ax) for ax in self.metadata.axes],
#                 compute=compute,
#                 name=self.name,
#             )
#         # Open existing store for updating labels only
#         elif isinstance(group, str):
#             group = zarr.open(group, mode="r+")

#         # write labels data if passed
#         if self.labels is not None:
#             labels_dict = cast(dict[str, OMEZarrMultiscale], self.labels)
#             label_group = group.require_group("labels")
#             for label_name, ms_labels in labels_dict.items():

#                 # coerce image name to name in labels dict
#                 ms_labels.name = label_name

#                 if label_name in label_group and not overwrite:
#                     warnings.warn(
#                         f"Label group {label_name} already exists in store. "
#                         f"Skipping writing this label since overwrite=False."
#                     )
#                     continue

#                 label_subgroup = label_group.require_group(label_name)

#                 # Write this label (always overwrite=True here since we've
#                 # already decided whether to skip based on the parent's flag)
#                 delayed += ms_labels.to_ome_zarr(
#                     group=label_subgroup,
#                     storage_options=storage_options,
#                     version=version,
#                     compute=compute,
#                     overwrite=True,
#                 )

#         list_of_labels = (
#             [str(label.name) for label in labels_dict.values()] if self.labels else []
#         )

#         # write the metadata to disk
#         if isinstance(group, str):
#             group = zarr.open(group, mode="r+")

#         # Only write full metadata if we wrote image data, otherwise just update labels
#         if write_image_data:
#             # Create a copy of metadata with normalized paths (s0, s1, etc.)
#             # to match the paths used by _write_pyramid_to_zarr
#             write_datasets = tuple(
#                 ds.model_copy(update={"path": f"s{idx}"})
#                 for idx, ds in enumerate(self.metadata.datasets)
#             )
#             write_metadata = self.metadata.model_copy(
#                 update={"datasets": write_datasets}
#             )

#             if version == "0.4":
#                 # in v0.4, metadata is stored under "multiscales" attribute
#                 metadata_dict = write_metadata.to_version("0.4").model_dump(
#                     by_alias=True
#                 )
#                 metadata_dict = _recursive_pop_nones(metadata_dict)
#                 metadata_dict["version"] = version
#                 group.attrs["multiscales"] = [metadata_dict]

#                 if self.omero and isinstance(self.omero, Omero):
#                     omero_dict = self.omero.model_dump(by_alias=True)
#                     omero_dict["version"] = version
#                     group.attrs["omero"] = omero_dict
#                 if self.image_label and isinstance(self.image_label, Label):
#                     image_label_dict = self.image_label.model_dump(by_alias=True)
#                     image_label_dict["version"] = version
#                     group.attrs["image-label"] = image_label_dict
#                 if list_of_labels:
#                     group_labels = group["labels"]
#                     group_labels.attrs["labels"] = list_of_labels

#             elif version == "0.5":
#                 metadata_dict = {
#                     "version": version,
#                     "multiscales": [
#                         _recursive_pop_nones(write_metadata.model_dump(by_alias=True))
#                     ],
#                 }

#                 if self.omero and isinstance(self.omero, Omero):
#                     omero_dict = self.omero.model_dump(by_alias=True)
#                     omero_dict["version"] = version
#                     metadata_dict["omero"] = omero_dict
#                 if self.image_label and isinstance(self.image_label, Label):
#                     image_label_dict = self.image_label.model_dump(by_alias=True)
#                     image_label_dict["version"] = version
#                     metadata_dict["image-label"] = image_label_dict

#                 if list_of_labels:
#                     group_labels = group["labels"]
#                     group_labels.attrs["ome"] = {
#                         "version": version,
#                         "labels": list_of_labels,
#                     }
#                 group.attrs["ome"] = metadata_dict

#         # Update mode: merge new labels with existing labels in metadata
#         elif list_of_labels:
#             group_labels = group["labels"]

#             if version == "0.4":
#                 existing_labels = group_labels.attrs.get("labels", [])
#                 # Merge: add new labels not already in existing list
#                 merged_labels = list(existing_labels)
#                 for label in list_of_labels:
#                     if label not in merged_labels:
#                         merged_labels.append(label)
#                 group_labels.attrs["labels"] = merged_labels
#             elif version == "0.5":
#                 existing_ome = group_labels.attrs.get("ome", {})
#                 existing_labels = existing_ome.get("labels", [])
#                 # Merge: add new labels not already in existing list
#                 merged_labels = list(existing_labels)
#                 for label in list_of_labels:
#                     if label not in merged_labels:
#                         merged_labels.append(label)
#                 group_labels.attrs["ome"] = {
#                     "version": version,
#                     "labels": merged_labels,
#                 }

#         return delayed

#     @classmethod
#     def from_ome_zarr(
#         cls,
#         group: zarr.Group | str,
#     ) -> OMEZarrMultiscale:
#         """
#         Load a multiscale pyramid from an OME-Zarr group.

#         Parameters
#         ----------
#         group : zarr.Group or str
#             The Zarr group or path containing the OME-Zarr data.

#         Returns
#         -------
#         NgffMultiscales`
#             A :class:`NgffMultiscales` container with the loaded images and metadata.
#         """
#         from ome_zarr.utils import _get_version

#         if isinstance(group, str):
#             group = zarr.open(group, mode="r")

#         version = _get_version(group)

#         list_of_labels = []
#         omero_dict = None
#         image_label_dict = None

#         # This is purely for backwards compatibility
#         # need to handle loading older metadata explicitly here
#         if version in ("0.1", "0.2", "0.3"):
#             metadata = cls._read_legacy_metadata(group, version)

#             if "labels" in group:
#                 labels_json = group["labels"].attrs.get("labels", [])
#                 list_of_labels = labels_json if isinstance(labels_json, list) else []
#             if "omero" in group.attrs:
#                 omero_dict = group.attrs.get("omero", None)
#             if "image-label" in group.attrs:
#                 image_label_dict = group.attrs.get("image-label", None)

#         elif version == "0.4":
#             from ome_zarr_models.v04.multiscales import Multiscale as Multiscalev04

#             metadata_json = cast(dict, group.attrs.get("multiscales", [None])[0])
#             metadata = Multiscalev04.model_validate(metadata_json).to_version("0.5")

#             if "labels" in group:
#                 labels_json = group["labels"].attrs.get("labels", [])
#                 list_of_labels = labels_json if isinstance(labels_json, list) else []
#             if "omero" in group.attrs:
#                 omero_dict = group.attrs.get("omero", None)
#             if "image-label" in group.attrs:
#                 image_label_dict = group.attrs.get("image-label", None)
#         elif version == "0.5":
#             from ome_zarr_models.v05.multiscales import Multiscale as Multiscalev05

#             ome_attrs = cast(dict, group.attrs.get("ome", {}))
#             metadata_json = ome_attrs.get("multiscales", [None])[0]
#             metadata = Multiscalev05.model_validate(metadata_json)

#             if "labels" in group:
#                 labels_ome_attrs = group["labels"].attrs.get("ome", {})
#                 list_of_labels = labels_ome_attrs.get("labels", [])

#             if "omero" in ome_attrs:
#                 omero_dict = ome_attrs.get("omero", None)
#             if "image-label" in ome_attrs:
#                 image_label_dict = ome_attrs.get("image-label", None)
#         else:
#             raise ValueError(f"Unsupported OME-Zarr version: {version}")

#         images = []
#         for dataset in metadata.datasets:
#             path = dataset.path
#             data = da.from_zarr(group[path])
#             scale = dataset.coordinateTransformations[0].scale
#             # Filter out axes with no unit, and set to None if empty
#             axes_units: dict[str, str] | None = {
#                 ax.name: ax.unit for ax in metadata.axes if ax.unit is not None
#             }
#             if not axes_units:
#                 axes_units = None
#             images.append(
#                 OMEZarrImage(
#                     data=data,
#                     axes=[ax.name for ax in metadata.axes],
#                     scale={ax.name: s for ax, s in zip(metadata.axes, scale)},
#                     axes_units=axes_units,
#                     name=str(metadata.name),
#                 )
#             )

#         # instantiate the NgffMultiscales object without calling __post_init__
#         instance = cls.__new__(cls)
#         instance.images = images
#         instance.metadata = metadata
#         instance.omero = None
#         instance.image_label = None
#         instance.labels = None
#         instance.method = None

#         # Finally, parse omero metadata
#         try:
#             if omero_dict is not None:
#                 instance.omero = Omero.model_validate(omero_dict)
#         except ValidationError as e:
#             warnings.warn(f"Invalid Omero metadata: {e}")

#         # parse image label metadata
#         try:
#             if image_label_dict is not None:
#                 instance.image_label = Label.model_validate(image_label_dict)
#         except ValidationError as e:
#             warnings.warn(f"Invalid image-label metadata: {e}")

#         if metadata.name is not None:
#             instance.name = metadata.name
#         else:
#             instance.name = "image"

#         # add labels if they exist
#         if list_of_labels:
#             labels = {}
#             for label_name in list_of_labels:
#                 label_group = group[f"labels/{label_name}"]
#                 label_multiscale = cls.from_ome_zarr(label_group)
#                 labels[label_name] = label_multiscale
#             instance.labels = labels

#         return instance

#     def __setattr__(self, name: str, value: Any) -> None:
#         """
#         Override setattr to ensure labels are always stored in dict format.
#         """
#         if name == "labels":
#             value = self._parse_labels(value)
#         elif name == "omero":
#             value = self._parse_omero(value)
#         elif name == "image_label":
#             value = self._parse_image_label(value)
#         super().__setattr__(name, value)

#     @staticmethod
#     def _parse_image_label(image_label: dict[str, Any] | Label | None) -> Label | None:
#         """
#         Helper method to coerce the `image_label` attribute to an instance
#         of the Label model for easier processing in other methods,
#         regardless of how it was originally passed by the user.
#         """
#         if image_label is None:
#             return None
#         elif isinstance(image_label, Label):
#             return image_label
#         elif isinstance(image_label, dict):
#             try:
#                 # We don't want to fail the entire initialization
#                 # if the image label metadata is invalid, so we
#                 # escape possible validation errors and just warn
#                 # the user that the image label metadata is invalid
#                 return Label.model_validate(image_label)
#             except ValidationError as e:
#                 warnings.warn(f"Invalid image-label metadata: {e}")
#                 return None
#         else:
#             raise ValueError(
#                 f"Invalid type for image_label: {type(image_label)}. "
#                 "Expected dict or Label instance."
#             )

#     @staticmethod
#     def _parse_omero(omero: dict[str, Any] | Omero | None) -> Omero | None:
#         """
#         Helper method to coerce the `omero` attribute to an instance
#         of the Omero model for easier processing in other methods,
#         regardless of how it was originally passed by the user.
#         """
#         if omero is None:
#             return None
#         elif isinstance(omero, Omero):
#             return omero
#         elif isinstance(omero, dict):
#             try:
#                 # We don't want to fail the entire initialization
#                 # if the omero metadata is invalid, so we
#                 # escape possible validation errors and just warn
#                 # the user that the omero metadata is invalid
#                 return Omero.model_validate(omero)
#             except ValidationError as e:
#                 warnings.warn(f"Invalid Omero metadata: {e}")
#                 return None
#         else:
#             raise ValueError(
#                 f"Invalid type for omero: {type(omero)}. "
#                 "Expected dict or Omero instance."
#             )

#     @staticmethod
#     def _parse_labels(
#         labels: (
#             OMEZarrMultiscale | list[OMEZarrMultiscale] | dict[str, OMEZarrMultiscale] | None
#         ),
#     ) -> dict[str, OMEZarrMultiscale] | None:
#         """
#         Helper method to coerce the `labels` attribute
#         to a consistent dict format for easier processing in other methods.,
#         regardless of how it was originally passed by the user.
#         """

#         if labels is None:
#             return None
#         elif isinstance(labels, OMEZarrMultiscale):
#             return {str(labels.name): labels}
#         elif isinstance(labels, list):
#             return {str(label.name): label for label in labels}
#         elif isinstance(labels, dict):
#             return labels
#         else:
#             raise ValueError(
#                 f"Invalid type for labels: {type(labels)}. "
#                 "Expected NgffMultiscales, list of NgffMultiscales, "
#                 "or dict of str to NgffMultiscales."
#             )

#     @staticmethod
#     def _read_legacy_metadata(group, version: str) -> MultiscaleV05:
#         from ome_zarr_models.v05.axes import Axis as AxisV05
#         from ome_zarr_models.v05.coordinate_transformations import (
#             VectorScale,
#             VectorTranslation,
#         )

#         metadata_json = cast(dict[str, Any], group.attrs.get("multiscales", [None])[0])

#         axes_map = {
#             "t": AxisV05(name="t", type="time"),
#             "c": AxisV05(name="c", type="channel"),
#             "z": AxisV05(name="z", type="space"),
#             "y": AxisV05(name="y", type="space"),
#             "x": AxisV05(name="x", type="space"),
#         }

#         axes_order: list[str] = ["t", "c", "z", "y", "x"]
#         if version == "0.3":
#             axes_order_value = metadata_json.get("axes")
#             if axes_order_value is None:
#                 raise ValueError(
#                     "Metadata version 0.3 requires 'axes' field in metadata"
#                 )
#             axes_order = cast(list[str], axes_order_value)

#         axes = [axes_map[ax] for ax in axes_order]

#         datasets = []
#         for idx, ds in enumerate(metadata_json.get("datasets", [])):
#             scale_level = [
#                 2.0 ** idx if s.name in ("z", "y", "x") else 1.0 for s in axes
#             ]

#             if idx == 0:
#                 transforms: tuple[VectorScale | VectorTranslation, ...] = (
#                     VectorScale(type="scale", scale=scale_level),
#                 )
#             else:
#                 translate = [
#                     2.0 ** (idx - 1) - 0.5 if s.name in ("z", "y", "x") else 0.0
#                     for s in axes
#                 ]
#                 transforms = (
#                     VectorScale(type="scale", scale=scale_level),
#                     VectorTranslation(type="translation", translation=translate),
#                 )

#             datasets.append(
#                 Dataset(
#                     path=ds.get("path", f"s{idx}"),
#                     coordinateTransformations=transforms,
#                 )
#             )

#         metadata = MultiscaleV05(
#             axes=axes,
#             datasets=tuple(datasets),
#             type=metadata_json.get("type", None),
#             metadata=metadata_json.get("metadata", None),
#             coordinateTransformations=None,
#             name=metadata_json.get("name", "image"),
#         )

#         return metadata
