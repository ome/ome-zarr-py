from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import dask.array as da
import numpy as np
import zarr
from yaozarrs import v05


class Methods(Enum):
    RESIZE = "resize"


SPATIAL_DIMS = ["z", "y", "x"]


@dataclass
class Image:
    data: da.Array | np.ndarray
    dims: Sequence[str] | str
    scale_factors: list[int] = field(default_factory=lambda: [2, 4, 8])
    scale: Sequence[float] | dict[str, float] | None = None
    scale_method: str | Methods = Methods.RESIZE
    axes_units: dict[str, str] | None = field(default_factory=dict)
    labels: dict[str, Any] | None = field(default_factory=dict)
    name: str | None = "image"

    multiscales: list["Image"] | None = None
    metadata: v05.Multiscale = field(init=False)
    _build_multiscales: bool = field(default=True, repr=False)

    def __post_init__(self):
        from .scale import _build_pyramid

        # set default scale if unset
        if not self.scale:
            self.scale = tuple(1.0 for s in range(len(self.dims)))

        # coerce dims to list of dims
        if isinstance(self.dims, str):
            self.dims = [d for d in self.dims]

        # coerce scale to dict if it's a sequence
        if isinstance(self.scale, Sequence):
            self.scale = {d: s for d, s in zip(self.dims, self.scale)}

        if isinstance(self.scale_method, Methods):
            self.scale_method = str(self.scale_method.value)

        datasets = [
            v05.Dataset(
                path="scale0",
                coordinateTransformations=[
                    v05.ScaleTransformation(scale=list(self.scale.values()))
                ],
            )
        ]

        axes = []
        for d in self.dims:
            if d in SPATIAL_DIMS:
                axes.append(v05.SpaceAxis(name=d))
            elif d == "t":
                axes.append(v05.TimeAxis(name=d))
            elif d == "c":
                axes.append(v05.ChannelAxis(name=d))

        self.metadata = v05.Multiscale(
            axes=axes,
            datasets=datasets,
            name=self.name,
        )

        if not self._build_multiscales:
            return

        pyramid = _build_pyramid(
            image=self.data,
            dims=self.dims,
            scale_factors=self.scale_factors,
            method=self.scale_method,
        )

        scales = [{d: self.scale[d] if d in SPATIAL_DIMS else 1 for d in self.dims}]
        for scale_factor in self.scale_factors:
            level_scale = {
                d: self.scale[d] * scale_factor if d in SPATIAL_DIMS else 1
                for d in self.dims
            }
            scales.append(level_scale)

        images = []
        datasets = []
        for idx, (level, scale) in enumerate(zip(pyramid, scales)):

            images.append(
                Image(
                    data=level,
                    dims=self.dims,
                    scale_factors=[],
                    scale=scale,
                    scale_method=self.scale_method,
                    axes_units=self.axes_units,
                    labels=self.labels,
                    name=self.name,
                    _build_multiscales=False,
                )
            )
            ds = v05.Dataset(
                path=f"scale{idx+1}",
                coordinateTransformations=[
                    v05.ScaleTransformation(scale=list(scale.values()))
                ],
            )
            datasets.append(ds)

        self.multiscales = images
        self.metadata = v05.Multiscale(
            axes=axes,
            datasets=datasets,
            name=self.name,
        )

    def to_ome_zarr(
        self,
        group: zarr.Group | str,
        version: str = "0.5",
    ):
        import os
        import shutil

        import zarr

        from .writer import write_multiscale

        if os.path.exists(str(group)):
            shutil.rmtree(str(group))

        write_multiscale(
            pyramid=[img.data for img in self.multiscales],
            group=group,
        )

        if isinstance(group, str):
            group = zarr.open(group, mode="r+")
            group.attrs["ome"] = self.metadata.model_dump(exclude_none=True)
