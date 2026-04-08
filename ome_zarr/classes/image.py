from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import InitVar, dataclass
from typing import Any, cast

import dask.array as da
import numpy as np
import zarr
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
from ome_zarr_models.v05.image_label_types import Label
from ome_zarr_models.v05.multiscales import (
    Dataset,
    Multiscale,
)
from pydantic import ValidationError

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
        if not self.scale:
            self.scale = tuple(1.0 for _ in range(len(self.axes)))

        # coerce axes to list
        if isinstance(self.axes, str):
            self.axes = list(self.axes)

        # coerce scale to dict if it's a sequence
        if isinstance(self.scale, Sequence):
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

    image: InitVar[NgffImage]
    scale_factors: InitVar[
        list[int] | tuple[int, ...] | list[dict[str, int]] | None
    ] = None
    method: str | Methods = Methods.RESIZE
    coordinateTransformations: InitVar[list[Scale | Translation | Identity] | None] = (
        None
    )
    labels: (
        NgffMultiscales | list[NgffMultiscales] | dict[str, NgffMultiscales] | None
    ) = None
    omero: dict[str, Any] | Omero | None = None
    image_label: dict[str, Any] | Label | None = None

    def __post_init__(
        self,
        image: NgffImage,
        scale_factors: list[int] | tuple[int, ...] | list[dict[str, int]] | None,
        coordinateTransformations: list[Scale | Translation | Identity] | None,
    ):
        from ome_zarr.scale import _build_pyramid

        if scale_factors is None:
            scale_factors = (2, 4, 8, 16)

        self.name = image.name
        method = self.method

        if isinstance(method, Methods):
            method = str(method.value)

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
                            scale=list(level_scale.values()),
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

        self.metadata = Multiscale(
            axes=tuple(axes),
            datasets=tuple(datasets),
            name=image.name,
            coordinateTransformations=coordinateTransformations,
        )

        # coerce labels to dict if it's a single NgffMultiscales or a list
        if self.labels is not None:
            if isinstance(self.labels, NgffMultiscales):
                self.labels = {str(self.labels.name): self.labels}
            elif isinstance(self.labels, list):
                self.labels = {str(label.name): label for label in self.labels}

        # parse omero metadata, if passed;
        # We don't want to fail the entire initialization if the omero metadata is invalid, so we
        # escape possible validation errors and just warn the user that the omero metadata is invalid
        try:
            if self.omero is None:
                return
            if isinstance(self.omero, Omero):
                return

            self.omero = Omero.model_validate(self.omero)
        except ValidationError as e:
            warnings.warn(f"Invalid Omero metadata: {e}")

        # parse image label metadata, if passed
        try:
            if self.image_label is None:
                return
            if isinstance(self.image_label, Label):
                return

            self.image_label = Label.model_validate(self.image_label)
        except ValidationError as e:
            warnings.warn(f"Invalid image-label metadata: {e}")

    def to_ome_zarr(
        self,
        group: zarr.Group | str,
        storage_options: list[dict[str, Any]] | dict[str, Any] | None = None,
        version: str | None = "0.5",
        compute: bool = True,
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
        fmt : Format, optional
            The OME-Zarr format version to use. Defaults to the current format (0.5).
        compute : bool, optional
            If True, compute immediately; otherwise return delayed objects.

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

        if os.path.exists(str(group)):
            shutil.rmtree(str(group))

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

        # write labels data if passed
        if self.labels is not None:
            labels_dict = cast(dict[str, NgffMultiscales], self.labels)
            for label_name, ms_labels in labels_dict.items():
                label_group = group.require_group(f"labels/{label_name}")

                delayed += ms_labels.to_ome_zarr(
                    group=label_group,
                    storage_options=storage_options,
                    version=version,
                    compute=compute,
                )

        list_of_labels = (
            [str(label.name) for label in labels_dict.values()] if self.labels else []
        )

        # write the metadata to disk
        if isinstance(group, str):
            group = zarr.open(group, mode="r+")

        if version == "0.4":
            # in v0.4, metadata is stored under "multiscales" attribute
            metadata_dict = self.metadata.to_version("0.4").model_dump()
            metadata_dict = _recursive_pop_nones(metadata_dict)

            if self.omero and isinstance(self.omero, Omero):
                metadata_dict["omero"] = self.omero.model_dump()
            if self.image_label and isinstance(self.image_label, Label):
                metadata_dict["image-label"] = self.image_label.model_dump()
            metadata_dict["version"] = version
            group.attrs["multiscales"] = [metadata_dict]

            if list_of_labels:
                group_labels = group["labels"]
                group_labels.attrs["labels"] = list_of_labels

        elif version == "0.5":
            metadata_dict = {
                "version": version,
                "multiscales": [_recursive_pop_nones(self.metadata.model_dump())],
            }
            if self.omero and isinstance(self.omero, Omero):
                metadata_dict["omero"] = self.omero.model_dump()
            if self.image_label and isinstance(self.image_label, Label):
                metadata_dict["image-label"] = self.image_label.model_dump()
            group.attrs["ome"] = metadata_dict

            if list_of_labels:
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
            from ome_zarr_models.v05.axes import Axis as AxisV05
            from ome_zarr_models.v05.coordinate_transformations import (
                VectorScale,
                VectorTranslation,
            )
            from ome_zarr_models.v05.multiscales import Multiscale as Multiscalev05

            metadata_json = cast(
                dict[str, Any], group.attrs.get("multiscales", [None])[0]
            )

            axes = (
                AxisV05(name="t", type="time"),
                AxisV05(name="c", type="channel"),
                AxisV05(name="z", type="space"),
                AxisV05(name="y", type="space"),
                AxisV05(name="x", type="space"),
            )

            scale = {s.name: 1.0 for s in axes}

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

            metadata = Multiscalev05(
                axes=axes,
                datasets=tuple(datasets),
                type=metadata_json.get("type", None),
                metadata=metadata_json.get("metadata", None),
                coordinateTransformations=None,
                name=metadata_json.get("name", "image"),
            )

        elif version == "0.4":
            from ome_zarr_models.v04.multiscales import Multiscale as Multiscalev04

            metadata_json = cast(dict, group.attrs.get("multiscales", [None])[0])
            metadata = Multiscalev04.model_validate(metadata_json).to_version("0.5")

            if "labels" in group:
                labels_json = group["labels"].attrs.get("labels", [])
                list_of_labels = labels_json if isinstance(labels_json, list) else []
            if "omero" in metadata_json:
                omero_dict = metadata_json.get("omero", None)
            if "image-label" in metadata_json:
                image_label_dict = metadata_json.get("image-label", None)
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
        else:
            raise ValueError(f"Unsupported OME-Zarr version: {version}")

        images = []
        for dataset in metadata.datasets:
            path = dataset.path
            data = da.from_zarr(group[path])
            scale = dataset.coordinateTransformations[0].scale
            # Filter out axes with no unit, and set to None if empty
            axes_units: dict[str, str] | None = {
                ax.name: ax.unit for ax in metadata.axes if ax.unit is not None
            }
            if not axes_units:
                axes_units = None
            images.append(
                NgffImage(
                    data=data,
                    axes=[ax.name for ax in metadata.axes],
                    scale={d.name: s for d, s in zip(metadata.axes, scale)},
                    axes_units=axes_units,
                    name=metadata.name,
                )
            )

        instance = cls.__new__(cls)
        instance.images = images
        instance.metadata = metadata

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
