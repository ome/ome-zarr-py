# View ome-zarr filesets
(tutorials:view_ome_zarr)=

To view ome-zarr datasets, you can use the [napari-ome-zarr](https://github.com/ome/napari-ome-zarr) plugin in the [napari](https://napari.org/) image viewer.

FOr instance, open any ome-zarr group from the CLI like this:

```bash
napari your_image.ome.zarr
```

This works irrespective of whether the data inside `your_image.ome.zarr`
is a single image or a plate containing multiple wells and fields.
The napari-ome-zarr plugin will automatically detect the structure of the data and display it accordingly:

```bash
napari your_hcs_data.ome.zarr
```

## From Python

You can also open an ome-zarr fileset from Python using the napari-ome-zarr plugin:

```python
import napari
viewer = napari.Viewer()
viewer.open("your_image.ome.zarr", plugin="napari-ome-zarr")
```