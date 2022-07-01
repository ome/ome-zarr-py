# 0.5.1 (June 2022)

- Read multi-resolution pyramids in `ome_zarr.reader.Well`, thanks to @tcompa ([#208](https://github.com/ome/ome-zarr-py/pull/208))

# 0.5.0 (June 2022)

- Add `fmt` option to `write_label_metadata` and store version ([#206](https://github.com/ome/ome-zarr-py/pull/206))

# 0.4.2 (May 2022)

- Relax version detection ([#189](https://github.com/ome/ome-zarr-py/pull/189))
- Fix format warning duplication ([#190](https://github.com/ome/ome-zarr-py/pull/190))
- Fix plate pyramid loading ([#195](https://github.com/ome/ome-zarr-py/pull/195))

# 0.4.1 (Mar 2022)

- Unify docstrings in `ome_zarr.writer` module ([#185](https://github.com/ome/ome-zarr-py/pull/185))

# 0.4.0 (Mar 2022)

- Add API for writing data according to the `image-label` spec: `write_labels`, `write_multiscale_labels` & `write_label_metadata` ([#178](https://github.com/ome/ome-zarr-py/pull/178), [#184](https://github.com/ome/ome-zarr-py/pull/184))
- Remove `byte_order` from `write_image` signature ([#184](https://github.com/ome/ome-zarr-py/pull/184))
- Deprecate `chunks` argument in favor of `storage_options` ([#184](https://github.com/ome/ome-zarr-py/pull/184))

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
