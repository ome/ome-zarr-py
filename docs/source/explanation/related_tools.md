# Related Tools

The OME-NGFF ecosystem includes many tools for creating, viewing, and analyzing OME-Zarr data.

## Viewers

### napari
[napari](https://napari.org/) is a fast, interactive multi-dimensional image viewer for Python.

- Install: `pip install napari[all]`
- Plugin: [napari-ome-zarr](https://github.com/ome/napari-ome-zarr)

```bash
napari your_image.zarr
```

### Neuroglancer
[Neuroglancer](https://github.com/google/neuroglancer) is a WebGL-based viewer for volumetric data.

### OME-NGFF Validator
[ome-ngff-validator](https://ome.github.io/ome-ngff-validator/) validates and previews OME-Zarr data in the browser.

## Conversion Tools

### bioformats2raw
[bioformats2raw](https://github.com/glencoesoftware/bioformats2raw) converts Bio-Formats supported files to OME-Zarr.

```bash
bioformats2raw input.tiff output.zarr
```

### ngff-zarr
[ngff-zarr](https://github.com/thewtex/ngff-zarr) is another Python library for OME-NGFF with ITK integration.

## Cloud Storage

### S3-compatible Storage
OME-Zarr works well with object storage:
- [Amazon S3](https://aws.amazon.com/s3/)
- [MinIO](https://min.io/) (self-hosted)
- [Cloudflare R2](https://www.cloudflare.com/products/r2/)

### fsspec
[fsspec](https://filesystem-spec.readthedocs.io/) provides unified filesystem access:

```python
import zarr
import s3fs

s3 = s3fs.S3FileSystem(anon=True)
store = s3fs.S3Map(root="bucket/path/image.zarr", s3=s3)
zarr_group = zarr.open(store, mode="r")
```

## Processing Libraries

### Dask
[Dask](https://dask.org/) enables parallel computing on larger-than-memory arrays.

### scikit-image
[scikit-image](https://scikit-image.org/) provides image processing algorithms.

### ITK
[ITK](https://itk.org/) is a comprehensive toolkit for image analysis.
