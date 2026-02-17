from __future__ import annotations

from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field
from typing import Any

import dask.array as da
import numpy as np
import zarr
from ome_zarr_models._v06.coordinate_transforms import (
  Scale,
  Transform,
  CoordinateSystem,
  CoordinateSystemIdentifier,
  Axis
)
from ome_zarr_models._v06.multiscales import (
    Dataset,
    Multiscale,
)

from .scale import Methods

SPATIAL_DIMS = ["z", "y", "x"]


@dataclass
class NgffImage:
    """
    Single-scale image representation with metadata.

    Parameters
    ----------
    data : dask.array.Array or numpy.ndarray
        The image data array.
    dims : sequence of str or str
        The dimension names corresponding to the data array axes, i.e. ('c', 'z', 'y', 'x').
    scale : sequence of float or dict of str to float, optional
        The physical scale for each dimension. If a sequence is provided, it should
        match the order of `dims`. If a dict is provided, keys should be dimension names,
        e.g. {'x': 0.1, 'y': 0.1, 'z': 0.5}. Default is None, which sets all scales to 1.0.
    axes_units : dict of str to str, optional
        Units for each dimension, e.g. {'x': 'micrometer', 'y': 'micrometer'}. Default is empty dict.
    name : str, optional
        Name of the image. Default is "image".

    Attributes
    ----------
    data : dask.array.Array
        The image data array.
    dims : sequence of str
        The dimension names.
    scale : dict of str to float
        The physical scale for each dimension.
    axes_units : dict of str to str
        Units for each dimension.
    name : str
        Name of the image.

    Methods
    -------
    to_multiscales(scale_factors=None, method=Methods.RESIZE) -> NgffMultiscales
        Build a multiscale pyramid from this image.
    """

    data: da.Array | np.ndarray
    dims: Sequence[str] | str
    scale: Sequence[float] | dict[str, float] | None = None
    axes_units: dict[str, str] | None = field(default_factory=dict)
    name: str | None = "image"

    def __post_init__(self):
        # set default scale if unset
        if not self.scale:
            self.scale = tuple(1.0 for _ in range(len(self.dims)))

        # coerce dims to list
        if isinstance(self.dims, str):
            self.dims = list(self.dims)

        # coerce scale to dict if it's a sequence
        if isinstance(self.scale, Sequence):
            self.scale = {d: s for d, s in zip(self.dims, self.scale)}

        # coerce data to dask array
        if not isinstance(self.data, da.Array):
            self.data = da.from_array(self.data)

        # validate dimensions match data shape
        if len(self.dims) != len(self.data.shape):
            raise ValueError(
                f"Number of dimensions in data ({len(self.data.shape)}) "
                f"does not match number of dims ({len(self.dims)})"
            )

    def to_multiscales(
        self,
        scale_factors: list[int] | None = None,
        method: str | Methods = Methods.RESIZE,
    ) -> NgffMultiscales:
        """
        Build a multiscale pyramid from this image.

        Parameters
        ----------
        scale_factors : list of int, optional
            Downsampling factors for each pyramid level. Default: [2, 4, 8].
        method : str or Methods, optional
            Downsampling method to use. Default: Methods.RESIZE.

        Returns
        -------
        Multiscales
            A Multiscales container with this image and its downsampled versions.
        """
        return NgffMultiscales.from_image(
            image=self,
            scale_factors=scale_factors,
            method=method,
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
        Downsampling factors for each pyramid level. Default: [2, 4, 8, 16].
    method : str or Methods, optional
        Downsampling method to use. Default: Methods.RESIZE.
    coordinate_system_name : str, optional
        Name of the coordinate system. Default: "physical".

    Attributes
    ----------
    images : list of NgffImage
        List of images at each pyramid level.
    metadata : Multiscale
        OME-Zarr multiscale metadata.

    Methods
    -------
    from_image(image, scale_factors=None, method=Methods.RESIZE) -> NgffMultiscales
        Build a multiscale pyramid from a base image.
    to_ome_zarr(group, storage_options=None, version="0.6", compute=True)
        Serialize the multiscale pyramid to an OME-Zarr group.
    from_ome_zarr(group) -> NgffMultiscales
        Load a multiscale pyramid from an OME-Zarr group.

    """
    image: InitVar[NgffImage]
    scale_factors: InitVar[list[int]] = [2, 4, 8, 16]
    method: str | Methods = Methods.RESIZE
    coordinate_system_name: InitVar[str | None] = "physical"
    coordinateTransformations: InitVar[list[Transform]] = []

    def __post_init__(
        self,
        image: NgffImage,
        scale_factors = [2, 4, 8, 16],
        coordinate_system_name: str | None = "physical",
        coordinateTransformations: list[Transform] = []
    ):
        from .scale import _build_pyramid

        self.name = image.name
        method = self.method
        if isinstance(method, Methods):
            method = str(method.value)

        if not coordinate_system_name:
            coordinate_system_name = "physical"

        # Build the pyramid data
        pyramid = _build_pyramid(
            image=image.data,
            dims=image.dims,
            scale_factors=scale_factors,
            method=method,
        )

        # build scales for each level based on the original image shape
        # and the pyramid level shapes
        scales = []
        for shape in [d.shape for d in pyramid]:
            scale = [full / level for full, level in zip(image.data.shape, shape)]
            scales.append({d: s * image.scale[d] for d, s in zip(image.dims, scale)})

        # Create Image instances for each pyramid level
        images = []
        datasets = []
        for idx, (level_data, level_scale) in enumerate(zip(pyramid, scales)):

            images.append(
                NgffImage(
                    data=level_data,
                    dims=image.dims,
                    scale=level_scale,
                    axes_units=image.axes_units,
                    name=image.name,
                )
            )
            datasets.append(
                Dataset(
                    path=f"scale{idx}",
                    coordinateTransformations=(
                        Scale(
                            input=f"scale{idx}",
                            output="physical",
                            scale=tuple(level_scale.values()),
                            path=None,
                        ),
                    ),
                )
            )

        self.images = images

        # Build axes metadata
        if image.axes_units is None:
            image.axes_units = {}

        axes = []
        for d in image.dims:
            if d in SPATIAL_DIMS:
                axes.append(Axis(name=d, type="space", unit=image.axes_units.get(d)))
            elif d == "t":
                axes.append(Axis(name=d, type="time", unit=image.axes_units.get(d)))
            elif d == "c":
                axes.append(Axis(name=d, type="channel", unit=image.axes_units.get(d)))
            else:
                axes.append(Axis(name=d, type="custom", unit=image.axes_units.get(d)))

        # check if any additional coordinate transforms have been passed and if so
        # add them to metadata and create a new output coordinate system
        coordinate_systems = []
        if coordinateTransformations:
            for tf in coordinateTransformations:
                if type(tf) is CoordinateSystemIdentifier:
                    name = tf.name
                else:
                    name = tf.output
                coordinate_systems.append(
                    CoordinateSystem(
                        name=name,
                        axes=tuple(
                            Axis(name=d.name, type=d.type, unit=d.unit) for d in axes
                        )
                    )
                )

        self.metadata = Multiscale(
            coordinateSystems=(
                CoordinateSystem(name=coordinate_system_name, axes=tuple(axes)),
                *coordinate_systems
            ),
            datasets=tuple(datasets),
            name=image.name,
            coordinateTransformations=tuple(coordinateTransformations),
        )

    @classmethod
    def from_image(
        cls,
        image: NgffImage,
        scale_factors: list[int] | None = None,
        method: str | Methods = Methods.RESIZE,
    ) -> NgffMultiscales:
        """
        Build a multiscale pyramid from a base image.

        Parameters
        ----------
        image : Image
            The base (highest resolution) image.
        scale_factors : list of int, optional
            Downsampling factors for each pyramid level. Default: [2, 4, 8].
        method : str or Methods, optional
            Downsampling method to use. Default: Methods.RESIZE.

        Returns
        -------
        Multiscales
            A Multiscales container with the pyramid images and metadata.
        """
        if scale_factors is None:
            scale_factors = [2, 4, 8]

        return cls(image=image, scale_factors=scale_factors, method=method)

    def to_ome_zarr(
        self,
        group: zarr.Group | str,
        storage_options: dict[str, Any] | None = None,
        version: str | None = "0.6",
        compute: bool = True,
    ):
        """
        Serialize the multiscale pyramid to an OME-Zarr group.

        Parameters
        ----------
        group : zarr.Group or str
            The target Zarr group or path where the OME-Zarr data will be written.
        storage_options : dict, optional
            Additional storage options to pass to Zarr, such as:
            - `compressor`: A Zarr compressor instance for compressing the data.
            - `chunks`: A tuple specifying the chunk shape for writing data.
        fmt : Format, optional
            The OME-Zarr format version to use. Defaults to the current format.
        compute : bool, optional
            If True, compute immediately; otherwise return delayed objects.
        """
        import os
        import shutil

        from .writer import _write_pyramid_to_zarr, check_group_fmt

        if os.path.exists(str(group)):
            shutil.rmtree(str(group))

        fmt = None
        if version == "0.6" or version == "0.5":
            from .format import FormatV05

            fmt = FormatV05()
        elif version == "0.4":
            from .format import FormatV04

            fmt = FormatV04()
        else:
            raise ValueError(f"Unsupported OME-Zarr version: {version}")

        group, fmt = check_group_fmt(group, fmt)

        # Coerce data to dask arrays for writing
        pyramid = [
            img.data if isinstance(img.data, da.Array) else da.from_array(img.data)
            for img in self.images
        ]

        _write_pyramid_to_zarr(
            pyramid=pyramid,
            group=group,
            storage_options=storage_options,
            fmt=fmt,
            axes=[dict(ax) for ax in self.metadata.coordinateSystems[0].axes],
            compute=compute,
        )

        if isinstance(group, str):
            group = zarr.open(group, mode="r+")

        if version == "0.4":
            # in v0.4, metadata is stored under "multiscales" attribute
            metadata_dict = self.metadata.to_version("0.4").model_dump()
            metadata_json = _recursive_pop_nones(metadata_dict)
            group.attrs["multiscales"] = [metadata_json]
        elif version in ("0.5", "0.6"):
            metadata_dict = {
                "version": version,
                "multiscales": [
                    _recursive_pop_nones(self.metadata.to_version(version).model_dump())
                ],
            }
            group.attrs["ome"] = metadata_dict

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
        NgffMultiscales
            A NgffMultiscales container with the loaded images and metadata.
        """

        if isinstance(group, str):
            group = zarr.open(group, mode="r")

        def _finditem(obj, key):
            if key in obj:
                return obj[key]
            for k, v in obj.items():
                if isinstance(v, dict):
                    item = _finditem(v, key)
                    if item is not None:
                        return item

        version = _finditem(group.attrs, "version")
        if version is None:
            raise ValueError("Could not find 'version' in group attributes")

        if version == "0.4":
            from ome_zarr_models._v04.multiscales import Multiscale as Multiscalev04

            metadata_json = group.attrs.get("multiscales", [None])[0]

            metadata = Multiscalev04.model_validate(metadata_json).to_version("0.6")
        elif version == "0.5":
            from ome_zarr_models._v05.multiscales import Multiscale as Multiscalev05

            ome_attrs = group.attrs.get("ome", {})
            metadata_json = ome_attrs.get("multiscales", [None])[0]
            metadata = Multiscalev05.model_validate(metadata_json).to_version("0.6")
        elif version == "0.6":
            from ome_zarr_models._v06.multiscales import Multiscale

            ome_attrs = group.attrs.get("ome", {})
            metadata_json = ome_attrs.get("multiscales", [None])[0]
            metadata_json = _recursive_pop_nones(metadata_json)
            metadata = Multiscale.model_validate(metadata_json)
        else:
            raise ValueError(f"Unsupported OME-Zarr version: {version}")

        images = []
        for dataset in metadata.datasets:
            path = dataset.path
            data = da.from_zarr(group[path])
            scale = dataset.coordinateTransformations[0].scale
            axes_units = {ax.name: ax.unit for ax in metadata.coordinateSystems[0].axes}
            if all(s is None for s in axes_units.values()):
                axes_units = None
            images.append(
                NgffImage(
                    data=data,
                    dims=[ax.name for ax in metadata.coordinateSystems[0].axes],
                    scale={
                        d.name: s
                        for d, s in zip(metadata.coordinateSystems[0].axes, scale)
                    },
                    axes_units=axes_units,
                    name=metadata.name,
                )
            )

        instance = cls.__new__(cls)
        instance.images = images
        instance.metadata = metadata
        return instance


def _recursive_pop_nones(input: dict) -> dict:
    """Recursively remove None values from a nested dictionary."""
    output = {}
    for key, value in input.items():
        if isinstance(value, dict):
            nested = _recursive_pop_nones(value)
            if nested:
                output[key] = nested
        elif isinstance(value, list | tuple):
            nested_list = []
            for item in value:
                if isinstance(item, dict):
                    nested_item = _recursive_pop_nones(item)
                    if nested_item:
                        nested_list.append(nested_item)
                elif item is not None:
                    nested_list.append(item)
            if nested_list:
                output[key] = nested_list
        elif value is not None:
            output[key] = value
    return output
