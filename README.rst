|pypi| |docs| |coverage|

===========
ome-zarr-py
===========

Tools for multi-resolution images stored in Zarr filesets, according to the `OME NGFF spec`_.

See `Readthedocs <https://ome-zarr.readthedocs.io/>`_ for usage information.

Documentation
-------------

Documentation will be automatically built with `readthedocs`.

It can be built locally with::

    $ pip install -r docs/requirements.txt
    $ sphinx-build -b html docs/source/ docs/build/html

Tests
-----

Tests can be run locally via `tox` with::

    $ pip install tox
    $ tox

To enable pre-commit code validation::

    $ pip install pre-commit
    $ pre-commit install

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
.. _`BSD`: https://opensource.org/licenses/BSD-2-Clause

.. |pypi| image:: https://badge.fury.io/py/ome-zarr.svg
    :alt: PyPI project
    :target: https://badge.fury.io/py/ome-zarr

.. |docs| image:: https://readthedocs.org/projects/ome-zarr/badge/?version=stable
    :alt: Documentation Status
    :target: https://ome-zarr.readthedocs.io/en/stable/?badge=stable

.. |coverage| image:: https://codecov.io/gh/ome/ome-zarr-py/branch/master/graph/badge.svg
    :alt: Test coverage
    :target: https://codecov.io/gh/ome/ome-zarr-py
