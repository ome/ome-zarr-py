# 0.10.3 (Janbuary 2025)

- Document Scaler attributes ([#418](https://github.com/ome/ome-zarr-py/pull/418))
- Fix dimension separator when downloading files ([#419](https://github.com/ome/ome-zarr-py/pull/419))
- Exclude bad version of fsspec in deps ([#420](https://github.com/ome/ome-zarr-py/pull/420))

# 0.10.2 (November 2024)

- Drop support for Python 3.8.
- Document parameters and return value of `parse_url`.
* Write metadata using delayed that depends on writing array(s)
* Document return values of sample data
* Simplify Python tutorial
* pin zarr at < 3
* Update version of pyupgrade

# 0.9.0 (May 2024)

- Correctly specify maximum compatible fsspec version. ([#338](https://github.com/ome/ome-zarr-py/pull/338))
- Add tests on Python 3.12. ([#338](https://github.com/ome/ome-zarr-py/pull/338))
- Write OMERO metadata. ([#261](https://github.com/ome/ome-zarr-py/pull/261))
- Fixed chunking when a scalar value for chunks is given. Previously
  passing ``storage_options={"chunks": <int>}`` only set the chunk
  size of the final dimension. Now the chunk size of all dimensions is
  set to this value, which is identical behaviour to ``zarr-python``.
  ([#365](https://github.com/ome/ome-zarr-py/pull/365))

# 0.8.3 (November 2023)

- Fix reading HCS file on AWS S3 ([#322](https://github.com/ome/ome-zarr-py/pull/322))
- Update docs to include write example for HCS dataset ([#317](https://github.com/ome/ome-zarr-py/pull/317))
- Minor improvements to format docs ([#316](https://github.com/ome/ome-zarr-py/pull/316))
- Add dask to intersphinx mapping ([#315](https://github.com/ome/ome-zarr-py/pull/315))

# 0.8.2 (September 2023)

- Fix compute on plate grid ([#299](https://github.com/ome/ome-zarr-py/pull/299))
- Check Well exists before trying to load data ([#296](https://github.com/ome/ome-zarr-py/pull/296))
- Avoid hard-coded field-of-view path "0". Thanks to [aeisenbarth](https://github.com/aeisenbarth) ([#300](https://github.com/ome/ome-zarr-py/pull/300))
- Update python.rst to have min, max for window. Thanks to [Sean Martin](https://github.com/seankmartin) ([#309](https://github.com/ome/ome-zarr-py/pull/309))
- Exclude fsspec 2023.9.0 fixes FileNotFoundError ([#307](https://github.com/ome/ome-zarr-py/pull/307))
- Set minimum fsspec version. Thanks to [Evan Lyall](https://github.com/elyall) ([#295](https://github.com/ome/ome-zarr-py/pull/295))
- Add conda install instructions to README. Thanks to [David Stansby](https://github.com/dstansby) ([#313](https://github.com/ome/ome-zarr-py/pull/313))
- Cody tidy fixes. Thanks to [Dimitri Papadopoulos Orfanos](https://github.com/DimitriPapadopoulos) ([#310](https://github.com/ome/ome-zarr-py/pull/310), [#312](https://github.com/ome/ome-zarr-py/pull/312))
# 0.8.0 (June 2023)

- Add `ome_zarr view image.zarr` to server and view in browser ([#285](https://github.com/ome/ome-zarr-py/pull/285))
- Allow `main` with empty args: thanks to Dominik Kutra ([#279](https://github.com/ome/ome-zarr-py/pull/279))

# 0.7.0 (May 2023)

- Optimize zarr multiscale writing by permitting delayed execution: thanks to Camilo Laiton ([#257](https://github.com/ome/ome-zarr-py/pull/257))
- Use default compressor by default; thanks to Juan Nunez-Iglesias ([#253](https://github.com/ome/ome-zarr-py/pull/253))
- Fix bug in dask resizing of edge chunks ([#244](https://github.com/ome/ome-zarr-py/pull/244))
- Fix dimension separator used in dask array to_zarr; thanks to Luca Marconato ([#243](https://github.com/ome/ome-zarr-py/pull/243))
- Add `compute` option to image writing methods; thanks to Camilo Laiton ([#257](https://github.com/ome/ome-zarr-py/pull/257))
- Allow errors to be raised when reading Multiscales data ([#266](https://github.com/ome/ome-zarr-py/pull/266))
- Add a pre-commit configuration for codespell; thanks to Yaroslav Halchenko ([#272](https://github.com/ome/ome-zarr-py/pull/272))

# 0.6.1 (October 2022)

- Fix pyramid dtype for scaler.local_mean ([#233](https://github.com/ome/ome-zarr-py/pull/233))
# 0.6.0 (September 2022)

- Add support for writing `dask` arrays. Thanks to [Daniel Toloudis](https://github.com/toloudis), with code contributions from [Andreas Eisenbarth](https://github.com/aeisenbarth) ([#192](https://github.com/ome/ome-zarr-py/pull/192))
- Fix a bug in using `storage_options` with `write_images`, thanks to [Marc Claesen](https://github.com/claesenm) ([#221](https://github.com/ome/ome-zarr-py/pull/221))
- Use multiscales name for node metadata ([#214](https://github.com/ome/ome-zarr-py/pull/214))

# 0.5.2 (July 2022)

- Disable auto_mkdir when path begins with s3, thanks to @colobas ([#212](https://github.com/ome/ome-zarr-py/pull/212))

# 0.5.1 (June 2022)

- Read multi-resolution pyramids in `ome_zarr.reader.Well`, thanks to @tcompa ([#209](https://github.com/ome/ome-zarr-py/pull/209))

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
