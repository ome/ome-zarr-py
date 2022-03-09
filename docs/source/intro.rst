===========
ome-zarr-py
===========

Experimental support for multi-resolution images stored in Zarr filesets, according to the `OME zarr spec`_.


Features
--------

- Use as a image reader plugin for `napari`_. See `napari-ome-zarr`_.
- Simple command-line to read and download conforming Zarr filesets.
- Helper methods for parsing related metadata.


Installation
------------

Install the latest release of `ome-zarr`_ from PyPI::

    pip install ome-zarr


Install developer mode to run from your current branch::

    git clone git@github.com:ome/ome-zarr-py.git
    cd ome-zarr-py
    pip install -e .


License
-------

Distributed under the terms of the `BSD`_ license,
"ome-zarr-py" is free and open source software

.. _`OME zarr spec`: https://github.com/ome/ngff
.. _`@napari`: https://github.com/napari
.. _`BSD`: https://opensource.org/licenses/BSD-2-Clause
.. _`Apache Software License 2.0`: http://www.apache.org/licenses/LICENSE-2.0
.. _`Mozilla Public License 2.0`: https://www.mozilla.org/media/MPL/2.0/index.txt
.. _`napari`: https://github.com/napari/napari
.. _`napari-ome-zarr`: https://github.com/ome/napari-ome-zarr
.. _`ome-zarr`: https://pypi.org/project/ome-zarr/
