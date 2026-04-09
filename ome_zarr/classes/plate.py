from dataclasses import dataclass
from typing import Any

from ome_zarr_models.common.well_types import WellImage, WellMeta
from ome_zarr_models.v05.plate import Acquisition, Column, Plate, Row, WellInPlate

from .image import NgffMultiscales


@dataclass
class NgffHCSPlate:
    """
    A plate in the HCS specification.
    """

    images: dict[tuple[str, str], list[NgffMultiscales]]

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

                # make sure we have a list of NgffMultiscales
            if not all(isinstance(item, NgffMultiscales) for item in value):
                raise TypeError(
                    f"Expected list of NgffMultiscales, got {type(value)} for key {key}"
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

        self.plate = Plate(
            rows=self.rows,
            columns=self.columns,
            wells=self.wells,
            acquisitions=[Acquisition(id=1, maximumfieldcount=1)],
        )

    def to_ome_zarr(
        self,
        group: zarr.Group | str,
        storage_options: list[dict[str, Any]] | dict[str, Any] | None = None,
        version: str = "0.5",
        compute: bool = True,
    ) -> list:

        import os
        import shutil

        from ome_zarr.format import Format, FormatV04, FormatV05
        from ome_zarr.utils import _recursive_pop_nones
        from ome_zarr.writer import check_group_fmt

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

        for key, images_in_well in self.images.items():
            well_group = group.require_group(f"{key[1]}/{key[0]}")
            well_images = []
            for i, image in enumerate(images_in_well):
                image_group = well_group.require_group(f"{i}")
                image.to_ome_zarr(
                    image_group,
                    storage_options=storage_options,
                    version=version,
                    compute=compute,
                )
                well_images.append(WellImage(acquisition=1, path=f"{i}"))

            well_metadata = WellMeta(images=well_images, version=version)

            if version == "0.5":
                well_group.attrs["ome"] = {
                    "well": _recursive_pop_nones(well_metadata.model_dump())
                }

        group.attrs["ome"] = {"plate": _recursive_pop_nones(self.plate.model_dump())}
