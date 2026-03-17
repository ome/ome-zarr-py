# ome-zarr-py

Tools for reading and writing multi-resolution images stored in Zarr filesets, according to the [OME NGFF spec](https://github.com/ome/ngff).

Note: The default version of OME-Zarr written by ``ome-zarr-py`` is ``v0.5``, which uses ``zarr v3``. OME-Zarr v0.5
is not yet supported by all OME-Zarr tools. See the documentation for more information on how to write other versions.

## Features

- {doc}`basic/cli_basics` for reading and downloading OME-ZARR filesets.
- {doc}`api` for reading and writing OME-ZARR filesets.
- Used by the [napari-ome-zarr](https://github.com/ome/napari-ome-zarr) plugin for viewing OME-ZARR filesets in [napari](https://github.com/napari/napari).


## Installation

Install the latest release of [`ome-zarr`](https://pypi.org/project/ome-zarr/) from PyPI:

```bash
pip install ome-zarr
```
or from conda-forge:

```bash
conda install -c conda-forge ome-zarr
```

## License

Distributed under the terms of the [BSD](https://opensource.org/licenses/BSD-2-Clause) license,
"ome-zarr-py" is free and open source software
