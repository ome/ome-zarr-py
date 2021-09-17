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
    $ ome_zarr info https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001240.zarr/

    # Local data (after downloading as below)
    $ ome_zarr info 6001240.zarr/

download
========

To download all the resolutions and metadata for an image::

    # creates local 6001240.zarr/
    $ ome_zarr download https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001240.zarr/

    # Specify output directory
    $ ome_zarr download https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001240.zarr/ --output image_dir

csv to labels
=============

The `csv_to_labels` command uses a CSV file to add key:value properties to labels
under an OME-Zarr Image or Plate.

The OME-Zarr labels metadata must already contain a `properties`
list of `{key:value}` objects, each with a unique key:ID. This key is `omero:shapeId`
in the example below.

This ID can be used to identify a single row of the CSV table by specifying the name of
a column with unique values, e.g. `shape_id` below.
This row is used to add additional column_name:value data to the label properties.

You also need to specify which columns from the CSV to use, e.g. `"area,X,Y,Width,Height"`.
You can also specify the column types (as in https://github.com/ome/omero-metadata/)
to specify the data-type for each column (string by default).

 - `d`: `DoubleColumn`, for floating point numbers
 - `l`: `LongColumn`, for integer numbers
 - `s`: `StringColumn`, for text
 - `b`: `BoolColumn`, for true/false

Use e.g. `#d` as a suffix in the column name to denote a `float` column, no spaces etc:
```
"area#d,label_text#s,Width#l,Height#l"
```

For example, to take values from columns named `area`, `label_text`, `Width` and `Height`
within a CSV file named `labels_data.csv` with an ID column named `shape_id` and add these
values to label properties with an ID key of `omero:shapeId` in an Image or Plate named `123.zarr`::

    ome_zarr csv_to_labels labels_data.csv shape_id "area#d,label_text#s,Width#l,Height#l" 123.zarr omero:shapeId```


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

.. _`OME zarr spec`: https://github.com/ome/ngff
.. _`@napari`: https://github.com/napari
.. _`BSD`: https://opensource.org/licenses/BSD-2-Clause
.. _`Apache Software License 2.0`: http://www.apache.org/licenses/LICENSE-2.0
.. _`Mozilla Public License 2.0`: https://www.mozilla.org/media/MPL/2.0/index.txt
.. _`napari`: https://github.com/napari/napari
.. _`napari-ome-zarr`: https://github.com/ome/napari-ome-zarr
.. _`ome-zarr`: https://pypi.org/project/ome-zarr/
