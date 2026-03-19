# View OME-ZARR images
(basic:view_images)=

A variety of tools exist to view ome-zarr images in different frameworks.

## Browser-based viewers

Web-based viewers are a simple way to access and view ome-zarr images located on a remote storage.
The [OME-NGFF-Validator](https://ome.github.io/ome-ngff-validator/) provides an entrypoint to
validation, introspection, and viewing of ome-zarr images in the browser:

<iframe width="100%" height="500" src="https://ome.github.io/ome-ngff-validator/"></iframe>

## Local viewers

Among the local viewers, [napari](https://napari.org/) is a popular choice for viewing and analyzing ome-zarr images in Python.
It requires the installation of the [napari-ome-zarr plugin](https://github.com/ome/napari-ome-zarr).

```bash
pip install napari napari-ome-zarr
```

To open any local or remote ome-zarr image, just pass the URL to napari:

```bash
napari https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr
```
