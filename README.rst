===========
ome-zarr-py
===========

Experimental support for images stored in Zarr filesets.


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

Use the `ome_zarr` command to interrogate and download Zarr datasets:

    $ ome_zarr info https://s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001240.zarr/
    $ ome_zarr download https://s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001240.zarr/
    $ ome_zarr info 6001240.zarr/

For example, to load select images by their ID in the Image Data Resource
such as http://idr.openmicroscopy.org/webclient/?show=image-6001240::

    $ napari 'https://s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001240.zarr/'

OR in python::

    import napari
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.open('https://s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001240.zarr/')


Alternatively, download one of these datasets with all associated metadata and
open locally::

    $ napari '/tmp/6001240.zarr/'

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

Distributed under the terms of the `GNU GPL v3.0`_ license,
"ome-zarr-py" is free and open source software


.. _`Cookiecutter`: https://github.com/audreyr/cookiecutter
.. _`@napari`: https://github.com/napari
.. _`GNU GPL v3.0`: http://www.gnu.org/licenses/gpl-3.0.txt
.. _`Apache Software License 2.0`: http://www.apache.org/licenses/LICENSE-2.0
.. _`Mozilla Public License 2.0`: https://www.mozilla.org/media/MPL/2.0/index.txt
.. _`cookiecutter-napari-plugin`: https://github.com/napari/cookiecutter-napari-plugin
.. _`napari`: https://github.com/napari/napari
.. _`ome-zarr`: https://pypi.org/project/ome-zarr/
