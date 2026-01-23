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
    scale: Sequence[float] | None = None
    scale_method: str | Methods = Methods.RESIZE
    axes_units: dict[str, str] | None = field(default_factory=dict)
    labels: dict[str, Any] | None = field(default_factory=dict)
    name: str | None = "image"

    multiscales: list["Image"] | None = None
    metadata: v05.Multiscale = field(init=False)
    _build_multiscales: bool = field(default=True, repr=False)

    def __post_init__(self):

        # set default scale if unset
        if not self.scale:
            self.scale = tuple(1.0 for s in range(len(self.dims)))

        # coerce dims to list of dims
        if isinstance(self.dims, str):
            self.dims = [d for d in self.dims]

        datasets = [
            v05.Dataset(
                path="s0",
                coordinateTransformations=[
                    v05.ScaleTransformation(scale=list(self.scale))
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

        # Create multiscales
        images = [
            Image(
                data=self.data,
                dims=self.dims,
                scale_factors=[],
                scale=self.scale,
                scale_method=self.scale_method,
                axes_units=self.axes_units,
                labels=self.labels,
                name=self.name,
                _build_multiscales=False,
            )
        ]

        for idx, factor in enumerate(self.scale_factors):

            # for scale factors except the root resolution, we use relative scale factors
            if idx == 0:
                relative_factor = int(self.scale_factors[0])
            else:
                relative_factor = self.scale_factors[idx] // self.scale_factors[idx - 1]

            relative_factor = tuple(
                relative_factor if d in SPATIAL_DIMS else 1 for d in self.dims
            )

            # Calculate target shape, leave non-spatial dims unchanged
            target_shape = [
                s // f if d in SPATIAL_DIMS else s
                for s, d, f in zip(images[-1].data.shape, self.dims, relative_factor)
            ]

            if self.scale_method == Methods.RESIZE:
                from .dask_utils import resize

                new_image = resize(images[-1].data, output_shape=tuple(target_shape))

            images.append(
                Image(
                    data=new_image,
                    dims=self.dims,
                    scale_factors=[],
                    scale=tuple(s * factor for s in self.scale),
                    scale_method=self.scale_method,
                    axes_units=self.axes_units,
                    labels=self.labels,
                    name=self.name,
                    _build_multiscales=False,
                )
            )
            ds = v05.Dataset(
                path=f"s{idx+1}",
                coordinateTransformations=[
                    v05.ScaleTransformation(scale=list(np.asarray(self.scale) * factor))
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
        group: zarr.Group,
        version: str = "0.5",
        chunks: tuple[Any, ...] | int | None = None,
    ):
        import os
        import shutil

        from .writer import write_multiscale

        if os.path.exists(str(group)):
            shutil.rmtree(str(group))

        write_multiscale(
            self.multiscales,
            group=group,
            chunks=chunks,
        )

        group.attrs["ome"] = self.metadata.model_dump(exclude_none=True)
