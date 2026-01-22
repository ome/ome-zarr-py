from dataclasses import dataclass, field
from typing import Any

import dask
import dask.array as da
import numpy as np
import zarr
from yaozarrs import v05

from .scale import Scaler


@dataclass
class Image:
    data: da.core.Array | np.ndarray
    dims: list[str] | str
    scale_factors: list[int] | None = field(default_factory=lambda: [2, 4, 8])
    scale: list[float] | None = None
    scale_method: str | None = "nearest"
    axes_units: dict[str, str] | None = field(default_factory=dict)
    labels: dict[str, Any] | None = field(default_factory=dict)
    name: str | None = "image"

    scaler: Scaler = field(init=False)
    multiscales: list = field(init=False)
    metadata: v05.Multiscale = field(init=False)

    def __post_init__(self):

        # set default scale if unset
        if not self.scale:
            self.scale = [1.0 for s in range(len(self.dims))]

        # coerce dims to list of dims
        if isinstance(self.dims, str):
            self.dims = [d for d in self.dims]

        # Create multiscales
        images = [self.data]
        datasets = [
            v05.Dataset(
                path="s0",
                coordinateTransformations=[v05.ScaleTransformation(scale=self.scale)],
            )
        ]

        for idx, factor in enumerate(self.scale_factors):

            # for scale factors except the root resolution, we use relative scale factors
            if idx == 0:
                relative_factor = int(self.scale_factors[0])
            else:
                relative_factor = self.scale_factors[idx] // self.scale_factors[idx - 1]

            spatial_dims = ["z", "y", "x"]
            target_shape = [
                s // relative_factor if d in spatial_dims else s
                for s, d in zip(images[-1].shape, self.dims)
            ]

            scale_function = Scaler(
                method=self.scale_method,
                downscale=relative_factor,
                max_layer=1,
                order=1,
            ).func

            new_image = da.from_delayed(
                dask.delayed(scale_function)(images[-1])[0],
                shape=target_shape,
                dtype=images[-1].dtype,
            )

            images.append(new_image)
            ds = v05.Dataset(
                path=f"s{idx+1}",
                coordinateTransformations=[
                    v05.ScaleTransformation(scale=list(np.asarray(self.scale) * factor))
                ],
            )
            datasets.append(ds)

        self.multiscales = images

        axes = []
        for d in self.dims:
            if d in spatial_dims:
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
