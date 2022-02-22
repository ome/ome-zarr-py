# 0.3.0 (Feb 2022)

## Specification changes

- Add support for [OME-NGFF 0.4](https://ngff.openmicroscopy.org/0.4/) specification ([#124](https://github.com/ome/ome-zarr-py/pull/124), [#159](https://github.com/ome/ome-zarr-py/pull/159), [#162](https://github.com/ome/ome-zarr-py/pull/162))

## API additions

- Add support for passing storage options (compression, chunks..) to the writer API, thanks to @satra ([#161](https://github.com/ome/ome-zarr-py/pull/161))
- Add API for writing plate & well metadata ([#153](https://github.com/ome/ome-zarr-py/pull/153), [#157](https://github.com/ome/ome-zarr-py/pull/157))
- Add API for writing multiscales metadata ([#149](https://github.com/ome/ome-zarr-py/pull/149), [#165](https://github.com/ome/ome-zarr-py/pull/165))

## Bug fix

- Fix remaining HCS assumptions that images are 5D ([#148](https://github.com/ome/ome-zarr-py/pull/148))
- Cap aiohttp to version 3.x ([#127](https://github.com/ome/ome-zarr-py/pull/127))

# 0.2.0 (Oct 2021)

## Breaking changes

- Add support for writing and scaling 2D, 3D and 4D data ([#114](https://github.com/ome/ome-zarr-py/pull/114))

## Infrastructure changes

- Add dispatch GitHub workflow ([#118](https://github.com/ome/ome-zarr-py/pull/118))

# 0.1.1 (Sep 2021)

## Bug fix

- Fix loading of HCS data with missing wells ([#111](https://github.com/ome/ome-zarr-py/pull/111))
- Point to new Embassy object store ([#109](https://github.com/ome/ome-zarr-py/pull/109))

# 0.1.0 (Sep 2021)
