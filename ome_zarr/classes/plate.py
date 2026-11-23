from dataclasses import dataclass
from typing import Any, Literal

import zarr
from ome_zarr_models.common.plate import Acquisition, Column, PlateBase, Row, WellInPlate
from ome_zarr_models.common.well_types import WellImage, WellMeta

from .image import OMEZarrMultiscale


@dataclass
class OMEZarrHCSPlate:
    """
    A plate in the HCS specification.

    ``images`` maps well paths "row/column", e.g. "A/1", to the images of that well.
    """

    images: dict[tuple[str, str], list[OMEZarrMultiscale]]

    def __post_init__(self):

        for (row_name, column_name), value in self.images.items():
            if not all(isinstance(item, OMEZarrMultiscale) for item in value):
                raise TypeError(
                    f"Expected list of OMEZarrMultiscale, got {type(value)} for key {(row_name, column_name)}"
                )

        self.metadata = self._populate_plate_metadata(self.images)

    def _populate_plate_metadata(self, images: dict[tuple[str, str], list[OMEZarrMultiscale]]) -> PlateBase:
        """
        Populate the metadata for the plate based on the provided images.
        This includes sorting the images, creating rows, columns, and wells,
        and generating the PlateBase metadata.
        """
        # sort images first by row (letter) then by column (number)
        self.images = dict(
            sorted(
                self.images.items(),
                key=lambda x: [int(p) if p.isdigit() else p for p in x[0]],
            )
        )
        self.rows = []
        self.columns = []
        self.wells = []
        for (row_name, column_name), value in self.images.items():
            if row_name not in self.rows:
                self.rows.append(row_name)
            if column_name not in self.columns:
                self.columns.append(column_name)

        # convert to ozmp instances
        self.rows = [Row(name=row) for row in self.rows]
        self.columns = [Column(name=column) for column in self.columns]

        # iterate over images again to find correct rowIndex and columnIndex for each well
        for (row_name, column_name), value in self.images.items():
            row_index = next(
                i for i, row in enumerate(self.rows) if row.name == row_name
            )
            column_index = next(
                i for i, column in enumerate(self.columns) if column.name == column_name
            )
            self.wells.append(
                WellInPlate(
                    path=f"{row_name}/{column_name}",
                    rowIndex=row_index,
                    columnIndex=column_index,
                )
            )

        return PlateBase(
                rows=self.rows,
                columns=self.columns,
                wells=self.wells,
                acquisitions=[Acquisition(id=1, maximumfieldcount=1)],
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
        if version in {"0.6", "0.5"}:
            fmt = FormatV05()
        elif version == "0.4":
            fmt = FormatV04()
        else:
            raise ValueError(f"Unsupported OME-Zarr version: {version}")

        group, fmt = check_group_fmt(group, fmt)

        delayed: list = []

        for key, images_in_well in self.images.items():
            well_group = group.require_group(f"{key[0]}/{key[1]}")
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

            well_attrs = WellMeta(version=version, images=well_images)

            if version in {"0.5", "0.6"}:
                metadata = {
                    "well": well_attrs.model_dump(exclude_none=True),
                    "version": str(version)
                }
                well_group.attrs.update({"ome": metadata})
            elif version in {"0.4"}:
                metadata = {
                    "well": well_attrs.model_dump(exclude_none=True),
                    "version": str(version)
                }
                well_group.attrs.update(metadata)

        if version in {"0.5", "0.6"}:
            metadata = {
                "plate": self.metadata.model_dump(exclude_none=True),
                "version": str(version)
            }
            group.attrs.update({"ome": metadata})
        elif version == "0.4":
            metadata = {
                "plate": self.metadata.model_dump(exclude_none=True),
                "version": str(version)
            }
            ome_attrs: dict = group.attrs if group.attrs else {}
            ome_attrs.update(metadata)

        return delayed
