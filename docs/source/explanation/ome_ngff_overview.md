# Understanding OME-NGFF

OME-NGFF (Open Microscopy Environment - Next Generation File Format) is a specification for storing
bioimaging data in a cloud-ready, analysis-friendly format based on [Zarr](https://zarr.dev/).

## What is OME-NGFF?

OME-NGFF defines conventions for storing multi-dimensional microscopy images with:

- **Multiscale pyramids**: Multiple resolution levels for efficient visualization at different zoom levels
- **Labeled axes**: Named dimensions (t, c, z, y, x) with types (time, channel, space)
- **Coordinate transformations**: Scale and translation metadata for physical coordinates
- **Labels/segmentations**: Associated segmentation masks stored alongside images
- **HCS plates**: High-content screening data with well-plate structures

## Resources

### Specification
- [OME-NGFF Specification](https://ngff.openmicroscopy.org/) - The official specification documents
- [GitHub Repository](https://github.com/ome/ngff) - Specification source and discussions

### Zarr Format
- [Zarr Documentation](https://zarr.readthedocs.io/) - The underlying storage format
- [Zarr v3 Specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html) - Latest zarr format

### Tools
- [napari-ome-zarr](https://github.com/ome/napari-ome-zarr) - Napari plugin for viewing OME-Zarr
- [ome-zarr-py](https://github.com/ome/ome-zarr-py) - This library
- [bioformats2raw](https://github.com/glencoesoftware/bioformats2raw) - Convert Bio-Formats supported files to OME-Zarr

### Community
- [Image.sc Forum](https://forum.image.sc/tag/ome-ngff) - Community discussions about OME-NGFF
- [OME Website](https://www.openmicroscopy.org/) - Open Microscopy Environment
