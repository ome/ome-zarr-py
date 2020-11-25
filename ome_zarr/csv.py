import csv
import os
from typing import Dict, Union

from zarr.convenience import open as zarr_open

from .io import parse_url

# d: DoubleColumn, for floating point numbers
# l: LongColumn, for integer numbers
# s: StringColumn, for text
# b: BoolColumn, for true/false
COLUMN_TYPES = ["d", "l", "s", "b"]


def parse_csv_value(value: str, col_type: str) -> Union[str, float, int, bool]:
    """Parse string value from csv, according to COLUMN_TYPES"""
    rv: Union[str, float, int, bool] = value
    try:
        if col_type == "d":
            rv = float(value)
        elif col_type == "l":
            rv = int(round(float(value)))
        elif col_type == "b":
            rv = bool(value)
    except ValueError:
        pass
    return rv


def csv_to_zarr(
    csv_path: str, csv_id: str, csv_keys: str, zarr_path: str, zarr_id: str
) -> None:
    """
    Add keys:values from a CSV file to the label properties of a Plate or Image

    For each labels properties dict in the Plate or Image at zarr_path, we pick the
    value of the property named zarr_id. This value should match a value from a row of
    the CSV table, with the column name given by csv_id. The row values under columns
    given by csv_keys (e.g. "col1,col2,col5") will be added to the label properties.
    Column types can be specified with #d (float), #l (int) or #b (boolean)
    e.g. "col1#d,col2#l,col5"

    @param csv_path         Path to the CSV file
    @param csv_id           Name of the CSV column to use as a row ID
    @param csv_keys         Names of the columns to add to properties.
    @param zarr_path        Path to the ome-zarr Plate that has labels with properties
    @param zarr_id          Key of the label property to use for picking csv row
    """

    # Use #d to denote double etc.
    # e.g. "area (pixels)#d,well_label#s,Width#l,Height#l"
    cols_types_by_name: Dict[str, str] = {}
    for col_name_type in csv_keys.split(","):
        if "#" in col_name_type:
            col_name, col_type = col_name_type.rsplit("#", 1)
            col_type = col_type if col_type in COLUMN_TYPES else "s"
            cols_types_by_name[col_name] = col_type
        else:
            cols_types_by_name[col_name_type] = "s"

    csv_columns = None
    id_column = None

    props_by_id: Dict[Union[str, int], Dict] = {}

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
            else:
                row_id = row[id_column]
                row_props = {}
                for col_name, value in zip(csv_columns, row):
                    if col_name in cols_types_by_name:
                        row_props[col_name] = parse_csv_value(
                            value, cols_types_by_name[col_name]
                        )
                props_by_id[row_id] = row_props

    dict_to_zarr(props_by_id, zarr_path, zarr_id)


def dict_to_zarr(
    props_to_add: Dict[Union[str, int], Dict], zarr_path: str, zarr_id: str
) -> None:
    """
    Add keys:values to the label properties of a ome-zarr Plate or Image.

    For each labels properties dict in the Plate or Image at zarr_path, we pick the
    value of the property named zarr_id. This value should match a key of the
    props_to_add Dict to find a Dict who's keys and values are added to the label
    properties.

    @param props_to_add     Dict of id: {key:value}. id matches values of zarr_id prop
    @param zarr_path        Path to the ome-zarr that has labels with properties
    @param zarr_id          Key of label property, where value is key of props_to_add
    """

    zarr = parse_url(zarr_path)
    if not zarr:
        raise Exception(f"No zarr found at {zarr_path}")

    plate_attrs = zarr.root_attrs.get("plate", None)
    multiscales = "multiscales" in zarr.root_attrs
    if plate_attrs is None and not multiscales:
        raise Exception("zarr_path must be to plate.zarr or image.zarr")

    labels_paths = []
    if plate_attrs is not None:
        # look for 'label/0' under the first field of each Well
        field = "0"
        for w in plate_attrs.get("wells", []):
            labels_paths.append(
                os.path.join(zarr_path, w["path"], field, "labels", "0")
            )
    else:
        labels_paths.append(os.path.join(zarr_path, "labels", "0"))

    for path_to_labels in labels_paths:

        label_group = zarr_open(path_to_labels)
        attrs = label_group.attrs.asdict()
        properties = attrs.get("image-label", {}).get("properties")
        if properties is None:
            continue

        for props_dict in properties:
            props_id = str(props_dict.get(zarr_id))

            # Look for key-value pairs to add to these properties
            if props_id in props_to_add:
                for key, value in props_to_add[props_id].items():
                    props_dict[key] = value

        label_group.attrs.update(attrs)
