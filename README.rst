===========
ome-zarr-py
===========

Experimental support for multi-resolution images stored in Zarr filesets, according to the `OME zarr spec`_.


Features
--------

- Use as a image reader plugin for `napari`_. The `napari`_ plugin was generated with `Cookiecutter`_ along with `@napari`_'s `cookiecutter-napari-plugin`_ template.
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


Usage
-----

Open Zarr filesets containing images with associated OME metadata.
The examples below use the image at http://idr.openmicroscopy.org/webclient/?show=image-6001240.

All examples can be made more or less verbose by passing `-v` or `-q` one or more times::

    # ome_zarr -vvv ...


info
====

Use the `ome_zarr` command to interrogate Zarr datasets::

    # Remote data
    $ ome_zarr info https://s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001240.zarr/

    # Local data (after downloading as below)
    $ ome_zarr info 6001240.zarr/

download
========

To download all the resolutions and metadata for an image::

    # creates local 6001240.zarr/
    $ ome_zarr download https://s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001240.zarr/

    # Specify output directory
    $ ome_zarr download https://s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001240.zarr/ --output image_dir

napari plugin
=============

Napari will use `ome-zarr` to open images that the plugin recognises as ome-zarr.
The image metadata from OMERO will be used to set channel names and rendering settings
in napari::

    $ napari 'https://s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001240.zarr/'

    # Also works with local files
    $ napari 6001240.zarr

OR in python::

    import napari
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.open('https://s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001240.zarr/')

If single zarray is passed to the plugin, it will be opened without the use of
the metadata::

    $ napari '/tmp/6001240.zarr/0'

Release process
---------------

This repository uses `bump2version <https://pypi.org/project/bump2version/>`_ to manage version numbers.
To tag a release run::

    $ bumpversion release

This will remove the ``.dev0`` suffix from the current version, commit, and tag the release.

To switch back to a development version run::

    $ bumpversion --no-tag [major|minor|patch]

specifying ``major``, ``minor`` or ``patch`` depending on whether the development branch will be a `major, minor or patch release <https://semver.org/>`_. This will also add the ``.dev0`` suffix.

Remember to ``git push`` all commits and tags.


License
-------

Distributed under the terms of the `BSD`_ license,
"ome-zarr-py" is free and open source software

.. _`OME zarr spec`: https://github.com/ome/omero-ms-zarr/blob/master/spec.md
.. _`Cookiecutter`: https://github.com/audreyr/cookiecutter
.. _`@napari`: https://github.com/napari
.. _`BSD`: https://opensource.org/licenses/BSD-2-Clause
.. _`Apache Software License 2.0`: http://www.apache.org/licenses/LICENSE-2.0
.. _`Mozilla Public License 2.0`: https://www.mozilla.org/media/MPL/2.0/index.txt
.. _`cookiecutter-napari-plugin`: https://github.com/napari/cookiecutter-napari-plugin
.. _`napari`: https://github.com/napari/napari
.. _`ome-zarr`: https://pypi.org/project/ome-zarr/
