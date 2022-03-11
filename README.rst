===========
ome-zarr-py
===========

Tools for multi-resolution images stored in Zarr filesets, according to the `OME NGFF spec`_.

See `Documentation <https://ome-zarr-py--121.org.readthedocs.build/en/121/>`_ for usage information.

Documentation
-------------

Documentation will be automatically built with `readthedocs`.

It can be built locally with:

    pip install spinx
    sphinx-build -b html docs/source/ docs/build/html

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

.. _`OME NGFF spec`: https://github.com/ome/ngff
.. _`@napari`: https://github.com/napari
.. _`BSD`: https://opensource.org/licenses/BSD-2-Clause
.. _`Apache Software License 2.0`: http://www.apache.org/licenses/LICENSE-2.0
.. _`Mozilla Public License 2.0`: https://www.mozilla.org/media/MPL/2.0/index.txt
.. _`napari`: https://github.com/napari/napari
.. _`napari-ome-zarr`: https://github.com/ome/napari-ome-zarr
.. _`ome-zarr`: https://pypi.org/project/ome-zarr/
