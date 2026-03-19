# Understanding OME-ZARR

OME-NGFF (Open Microscopy Environment - Next Generation File Format) is a specification for storing
bioimaging data in a cloud-ready, analysis-friendly format based on [Zarr](https://zarr.dev/).

## What is OME-ZARR?

OME-ZARR defines conventions for storing multi-dimensional microscopy images with:

- **Multiscale pyramids**: Multiple resolution levels for efficient visualization at different zoom levels
- **Labeled axes**: Named dimensions (t, c, z, y, x) with types (time, channel, space)
- **Coordinate transformations**: Scale and translation metadata for physical coordinates
- **Labels/segmentations**: Associated segmentation masks stored alongside images
- **HCS plates**: High-content screening data with well-plate structures

## Resources

### Specification
- [OME-NGFF Specification](https://ngff.openmicroscopy.org/specifications) - The official specification document
- [Specification GitHub Repository](https://github.com/ome/ngff-spec) - Specification source
- [NGFF website repository](https://github.com/ome/ngff) - Source for NGFF website and place for discussions about the format


### Zarr Format
- [Zarr Documentation](https://zarr.readthedocs.io/) - The underlying storage format
- [Zarr v3 Specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html) - Latest zarr format

### Tools

A comprehensive list of tools to generate, convert, view or work with ome-zarr images in general
is provided on the [ngff website](https://ngff.openmicroscopy.org/resources/tools/index.html)

### Community
- [Image.sc Forum](https://forum.image.sc/tag/ome-ngff) - Community discussions about OME-NGFF
- [OME Website](https://www.openmicroscopy.org/) - Open Microscopy Environment
