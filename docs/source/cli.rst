.. highlight:: bash


Command-line tool
-----------------

Open Zarr filesets containing images with associated OME metadata.
The examples below use the image at http://idr.openmicroscopy.org/webclient/?show=image-6001240.

All examples can be made more or less verbose by passing `-v` or `-q` one or more times::

    ome_zarr -vvv ...


info
====

Use the `ome_zarr` command to interrogate Zarr datasets.

Remote data::

    ome_zarr info https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001240.zarr/

Local data::

    ome_zarr info 6001240.zarr/

view
====

Use the `ome_zarr` command to view Zarr data in the https://ome.github.io/ome-ngff-validator::

    ome_zarr view 6001240.zarr/

download
========

To download all the resolutions and metadata for an image use ``ome_zarr download``. This creates ``6001240.zarr`` locally::

    ome_zarr download https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001240.zarr/

Specify a different output directory::

    ome_zarr download https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001240.zarr/ --output image_dir

create
======

To create sample OME-Zarr image from the `skimage <https://scikit-image.org/docs/stable/api/skimage.data.html>`_
data.

Create an OME-Zarr image in coinsdata/ dir::

    ome_zarr create coinsdata

Create an rgb image from skimage astronaut in testimage dir::

    ome_zarr create testimage --method=astronaut

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

Use e.g. `#d` as a suffix in the column name to denote a `float` column, no spaces etc.::

    "area#d,label_text#s,Width#l,Height#l"


For example, to take values from columns named `area`, `label_text`, `Width` and `Height`
within a CSV file named `labels_data.csv` with an ID column named `shape_id` and add these
values to label properties with an ID key of `omero:shapeId` in an Image or Plate named `123.zarr`::

    ome_zarr csv_to_labels labels_data.csv shape_id "area#d,label_text#s,Width#l,Height#l" 123.zarr omero:shapeId
