# Multiscale Pyramids

Multiscale image pyramids are a fundamental concept in OME-NGFF that enable efficient
visualization and analysis of large images.

## Why Pyramids?

Modern microscopy produces images that can be gigabytes or even terabytes in size.
Loading an entire image at full resolution is:

- **Slow**: Transferring large amounts of data takes time
- **Memory-intensive**: May exceed available RAM
- **Unnecessary**: When viewing zoomed out, full resolution is wasteful

## How Pyramids Work

A pyramid stores the same image at multiple resolution levels:

```
Level 0: 4096 x 4096  (full resolution)
Level 1: 2048 x 2048  (2x downsampled)
Level 2: 1024 x 1024  (4x downsampled)
Level 3:  512 x  512  (8x downsampled)
```

Viewers load only the resolution level appropriate for the current zoom level,
enabling smooth navigation of arbitrarily large images.

## Downsampling Methods

Different downsampling methods are appropriate for different data types:

| Method | Use Case |
|--------|----------|
|`ome_zarr.scale.Methods.resize` | Fast skimage-based resizing |
| `ome_zarr.scale.Methods.nearest` | Categorical data (labels, segmentations) |
| `ome_zarr.scale.Methods.zoom` | Downsampling using the [scipy zoom function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html#scipy.ndimage.zoom) |
| `ome_zarr.scale.Methods.local_mean` | Local averaging for smoother results |

See {py:class}`ome_zarr.scale.Methods` for all available options and more details.

## Resources

- [Zarr Chunking Guide](https://zarr.readthedocs.io/en/stable/user-guide/performance/#chunk-optimizations)
- [Dask Array Documentation](https://docs.dask.org/en/stable/array.html)
