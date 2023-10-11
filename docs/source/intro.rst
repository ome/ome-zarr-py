===========
ome-zarr-py
===========

Tools for reading and writing multi-resolution images stored in Zarr filesets, according to the `OME NGFF spec`_.


Features
--------

- :doc:`cli` for reading and downloading OME-NGFF filesets.
- :doc:`python` for reading and writing OME-NGFF filesets.
- `ome-zarr-py` is used by the `napari-ome-zarr`_ plugin for viewing OME-NGFF filesets in `napari`.


Installation
------------

Install the latest release of `ome-zarr`_ from PyPI::

    pip install ome-zarr

or from conda-forge::

    conda install -c conda-forge ome-zarr

Installation for developers::

    git clone git@github.com:ome/ome-zarr-py.git
    cd ome-zarr-py
    pip install -e .


License
-------

Distributed under the terms of the `BSD`_ license,
"ome-zarr-py" is free and open source software

.. _`OME NGFF spec`: https://github.com/ome/ngff
.. _`@napari`: https://github.com/napari
.. _`BSD`: https://opensource.org/licenses/BSD-2-Clause
.. _`Apache Software License 2.0`: http://www.apache.org/licenses/LICENSE-2.0
.. _`Mozilla Public License 2.0`: https://www.mozilla.org/media/MPL/2.0/index.txt
.. _`napari`: https://github.com/napari/napari
.. _`napari-ome-zarr`: https://github.com/ome/napari-ome-zarr
.. _`ome-zarr`: https://pypi.org/project/ome-zarr/
