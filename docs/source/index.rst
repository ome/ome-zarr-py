===========
ome-zarr-py
===========

Tools for reading and writing multi-resolution images stored in Zarr filesets, according to the `OME NGFF spec`_.


Features
--------

- :doc:`cli` for reading and downloading OME-NGFF filesets.
- :doc:`api` for reading and writing OME-NGFF filesets (see :doc:`python` for example usage).
- Used by the `napari-ome-zarr`_ plugin for viewing OME-NGFF filesets in `napari`_.

Contents
--------
.. toctree::
   :maxdepth: 1

   cli
   python
   api

Installation
------------

Install the latest release of `ome-zarr`_ from PyPI::

    pip install ome-zarr

or from conda-forge::

    conda install -c conda-forge ome-zarr


License
-------

Distributed under the terms of the `BSD`_ license,
"ome-zarr-py" is free and open source software

.. _`OME NGFF spec`: https://github.com/ome/ngff
.. _`BSD`: https://opensource.org/licenses/BSD-2-Clause
.. _`napari`: https://github.com/napari/napari
.. _`napari-ome-zarr`: https://github.com/ome/napari-ome-zarr
.. _`ome-zarr`: https://pypi.org/project/ome-zarr/
