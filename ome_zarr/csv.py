import csv
import os
from typing import Dict

from zarr.convenience import open as zarr_open

from .io import parse_url


def csv_to_zarr(
    csv_path: str, csv_id: str, csv_keys: str, zarr_path: str, zarr_id: str
) -> None:
    """
    - URL of the form: path/to/ID.zarr/
    """

    cols_to_read = csv_keys.split(",")

    csv_columns = None
    id_column = None

    props_by_id: Dict[str, Dict] = {}

    with open(csv_path, newline="") as csvfile:
        row_reader = csv.reader(csvfile, delimiter=",")
        for row in row_reader:
            # For the first row, find the Column that has the ID
            if csv_columns is None:
                csv_columns = row
                if csv_id not in csv_columns:
                    raise ValueError(
                        f"csv_id '{csv_id}' should match a"
                        f"csv column name: {csv_columns}"
                    )
                id_column = csv_columns.index(csv_id)
                print("id_column", id_column)
            else:
                row_id = row[id_column]
                row_props = {}
                for col_name, value in zip(csv_columns, row):
                    if col_name in cols_to_read:
                        row_props[col_name] = value
                props_by_id[row_id] = row_props

    dict_to_zarr(props_by_id, zarr_path, zarr_id)


def dict_to_zarr(props_to_add: Dict, zarr_path: str, zarr_id: str) -> None:

    zarr = parse_url(zarr_path)
    print("zarr", zarr)
    if not zarr:
        print(f"No zarr found at {zarr_path}")
        return

    print(zarr.root_attrs)
    plate_attrs = zarr.root_attrs.get("plate", None)
    if plate_attrs is None:
        print(f"zarr_path must be to plate.zarr. No 'plate' in {zarr_path}.zattrs")
    well_paths = [w["path"] for w in plate_attrs.get("wells", [])]

    # look for 'label/0' under the first field of each Well
    field = "0"
    for well in well_paths:
        path_to_labels = os.path.join(zarr_path, well, field, "labels", "0")
        print("path_to_labels", path_to_labels)

        label_group = zarr_open(path_to_labels)
        attrs = label_group.attrs.asdict()
        properties = attrs.get("image-label", {}).get("properties")
        if properties is None:
            continue

        print("attrs", attrs)
        attrs["test"] = True

        for props_dict in properties:
            props_id = str(props_dict.get(zarr_id))
            print("props_id", props_id)

            # Look for key-value pairs to add to these properties
            if props_id in props_to_add:
                for key, value in props_to_add[props_id].items():
                    props_dict[key] = value

        label_group.attrs.update(attrs)
