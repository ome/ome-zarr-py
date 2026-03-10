# Write HCS Plates
(tutorials:write_hcs_plate)=

This tutorial shows how to write a high-content screening (HCS) dataset to OME-NGFF format.
HCS datasets represent culture plates with multiple wells, where each well can contain multiple fields of view.

## Create sample data

First, let's set up some sample data representing a multi-well plate:

```python
import numpy as np
import zarr

from ome_zarr.writer import write_image, write_plate_metadata, write_well_metadata

path = "test_ngff_plate.zarr"
row_names = ["A", "B"]
col_names = ["1", "2", "3"]
well_paths = ["A/2", "B/3"]
field_paths = ["0", "1", "2"]

# generate data
mean_val = 10
num_wells = len(well_paths)
num_fields = len(field_paths)
size_xy = 128
size_z = 10
rng = np.random.default_rng(0)
data = rng.poisson(mean_val, size=(num_wells, num_fields, size_z, size_xy, size_xy)).astype(np.uint8)
```

## Write plate structure

The plate is written by creating the hierarchical zarr structure with plate and well metadata:

```python
# write the plate of images and corresponding metadata
# Use zarr_format=2 to write v0.4 format (zarr v2)
root = zarr.open_group(path, mode="w")
write_plate_metadata(root, row_names, col_names, well_paths)

for wi, wp in enumerate(well_paths):
    row, col = wp.split("/")
    row_group = root.require_group(row)
    well_group = row_group.require_group(col)
    write_well_metadata(well_group, field_paths)
    for fi, field in enumerate(field_paths):
        image_group = well_group.require_group(str(field))
        write_image(image=data[wi, fi], group=image_group, axes="zyx",
                    storage_options=dict(chunks=(1, size_xy, size_xy)))
```

## View the plate

This plate can be viewed in `napari` using the
[`napari-ome-zarr`](https://github.com/ome/napari-ome-zarr) plugin:

```python
import napari

viewer = napari.Viewer()
viewer.open(path, plugin="napari-ome-zarr")
```

Or from the command line:

```bash
napari test_ngff_plate.zarr
```
