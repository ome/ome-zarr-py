from dataclasses import dataclass
from typing import Any, Literal

import zarr
from ome_zarr_models.v06.hcs import HCSAttrs
from ome_zarr_models.v06.plate import Acquisition, Column, Plate, Row, WellInPlate
from ome_zarr_models.v06.well import WellAttrs
from ome_zarr_models.v06.well_types import WellImage, WellMeta

from .image import OMEZarrMultiscale


@dataclass
class OMEZarrHCSPlate:
    """
    A plate in the HCS specification.
    """

    images: dict[tuple[str, str], list[OMEZarrMultiscale]]

    def __post_init__(self):

        # sort images first by row (letter) then by column (number)
        self.images = dict(
            sorted(
                self.images.items(),
                key=lambda x: (x[0][1], int(x[0][0]) if x[0][0].isdigit() else x[0][0]),
            )
        )
        self.rows = []
        self.columns = []
        self.wells = []
        for key, value in self.images.items():
            if key[1] not in self.rows:
                self.rows.append(key[1])
            if key[0] not in self.columns:
                self.columns.append(key[0])

                # make sure we have a list of OMEZarrMultiscale
            if not all(isinstance(item, OMEZarrMultiscale) for item in value):
                raise TypeError(
                    f"Expected list of OMEZarrMultiscale, got {type(value)} for key {key}"
                )

        # convert to ozmp instances
        self.rows = [Row(name=row) for row in self.rows]
        self.columns = [Column(name=column) for column in self.columns]

        # iterate over images again to find correct rowIndex and columnIndex for each well
        for key, value in self.images.items():
            row_index = next(i for i, row in enumerate(self.rows) if row.name == key[1])
            column_index = next(
                i for i, column in enumerate(self.columns) if column.name == key[0]
            )
            self.wells.append(
                WellInPlate(
                    path=f"{key[1]}/{key[0]}",
                    rowIndex=row_index,
                    columnIndex=column_index,
                )
            )

        self._metadata = HCSAttrs(
            version="0.6",
            plate=Plate(
                rows=self.rows,
                columns=self.columns,
                wells=self.wells,
                acquisitions=[Acquisition(id=1, maximumfieldcount=1)],
            ),
        )

    def to_ome_zarr(
        self,
        group: zarr.Group | str,
        storage_options: list[dict[str, Any]] | dict[str, Any] | None = None,
        version: Literal["0.4", "0.5", "0.6"] = "0.6",
        overwrite: bool = False,
        compute: bool = True,
    ) -> list:

        import os
        import shutil

        from ome_zarr.format import Format, FormatV04, FormatV05
        from ome_zarr.writer import check_group_fmt

        if os.path.exists(str(group)):
            if overwrite:
                shutil.rmtree(str(group))
            else:
                raise FileExistsError(
                    f"Group {group} already exists and overwrite is False."
                )

        fmt: Format | None = None
        if version == "0.6" or version == "0.5":
            fmt = FormatV05()
        elif version == "0.4":
            fmt = FormatV04()
        else:
            raise ValueError(f"Unsupported OME-Zarr version: {version}")

        group, fmt = check_group_fmt(group, fmt)

        delayed: list = []

        for key, images_in_well in self.images.items():
            well_group = group.require_group(f"{key[1]}/{key[0]}")
            well_images = []
            for i, image in enumerate(images_in_well):
                image_group = well_group.require_group(str(i))
                delayed += image.to_ome_zarr(
                    image_group,
                    storage_options=storage_options,
                    version=version,
                    overwrite=overwrite,
                    compute=compute,
                )
                well_images.append(WellImage(acquisition=1, path=f"{i}"))

            well_attrs = WellAttrs(version=version, well=WellMeta(images=well_images))
            well_group.attrs["ome"] = well_attrs.model_dump(exclude_none=True)

        ome_attrs: dict = group.attrs.get("ome", {})
        ome_attrs.update(self._metadata.model_dump(exclude_none=True))
        group.attrs["ome"] = ome_attrs

        return delayed
