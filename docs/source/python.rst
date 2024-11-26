Python tutorial
===============

Writing OME-NGFF images
-----------------------

The principle entry-point for writing OME-NGFF images is :py:func:`ome_zarr.writer.write_image`.
This takes an n-dimensional `numpy` array or `dask` array and writes it to the specified `zarr group` according
to the OME-NGFF specification.
By default, a pyramid of resolution levels will be created by down-sampling the data by a factor
of 2 in the X and Y dimensions.

Alternatively, the :py:func:`ome_zarr.writer.write_multiscale` can be used, which takes a
"pyramid" of pre-computed `numpy` arrays.

The following code creates a 3D Image in OME-Zarr::

    import numpy as np
    import zarr

    from ome_zarr.io import parse_url
    from ome_zarr.writer import write_image

    path = "test_ngff_image.zarr"

    size_xy = 128
    size_z = 10
    rng = np.random.default_rng(0)
    data = rng.poisson(lam=10, size=(size_z, size_xy, size_xy)).astype(np.uint8)

    # write the image data
    store = parse_url(path, mode="w").store
    root = zarr.group(store=store)
    write_image(image=data, group=root, axes="zyx", storage_options=dict(chunks=(1, size_xy, size_xy)))


This image can be viewed in `napari` using the
`napari-ome-zarr <https://github.com/ome/napari-ome-zarr>`_ plugin::

    $ napari test_ngff_image.zarr

Rendering settings
------------------
Render settings can be added to an existing zarr group::

    store = parse_url(path, mode="w").store
    root = zarr.group(store=store)
    root.attrs["omero"] = {
        "channels": [{
            "color": "00FFFF",
            "window": {"start": 0, "end": 20, "min": 0, "max": 255},
            "label": "random",
            "active": True,
        }]
    }

Writing labels
--------------

The following code creates a 3D Image in OME-Zarr with labels::

    import numpy as np
    import zarr
    import os

    from skimage.data import binary_blobs
    from ome_zarr.io import parse_url
    from ome_zarr.writer import write_image

    path = "test_ngff_image.zarr"
    os.mkdir(path)

    mean_val=10
    size_xy = 128
    size_z = 10
    rng = np.random.default_rng(0)
    data = rng.poisson(mean_val, size=(size_z, size_xy, size_xy)).astype(np.uint8)

    # write the image data
    store = parse_url(path, mode="w").store
    root = zarr.group(store=store)
    write_image(image=data, group=root, axes="zyx", storage_options=dict(chunks=(1, size_xy, size_xy)))
    # optional rendering settings
    root.attrs["omero"] = {
        "channels": [{
            "color": "00FFFF",
            "window": {"start": 0, "end": 20, "min": 0, "max": 255},
            "label": "random",
            "active": True,
        }]
    }


    # add labels...
    blobs = binary_blobs(length=size_xy, volume_fraction=0.1, n_dim=3).astype('int8')
    blobs2 = binary_blobs(length=size_xy, volume_fraction=0.1, n_dim=3).astype('int8')
    # blobs will contain values of 1, 2 and 0 (background)
    blobs += 2 * blobs2

    # label.shape is (size_xy, size_xy, size_xy), Slice to match the data
    label = blobs[:size_z, :, :]

    # write the labels to /labels
    labels_grp = root.create_group("labels")
    # the 'labels' .zattrs lists the named labels data
    label_name = "blobs"
    labels_grp.attrs["labels"] = [label_name]
    label_grp = labels_grp.create_group(label_name)
    # need 'image-label' attr to be recognized as label
    label_grp.attrs["image-label"] = {
        "colors": [
            {"label-value": 1, "rgba": [255, 0, 0, 255]},
            {"label-value": 2, "rgba": [0, 255, 0, 255]},
            {"label-value": 3, "rgba": [255, 255, 0, 255]}
        ]
    }

    write_image(label, label_grp, axes="zyx")

Writing HCS datasets to OME-NGFF
--------------------------------

This sample code shows how to write a high-content screening dataset (i.e. culture plate with multiple wells) to a OME-NGFF file::

    import numpy as np
    import zarr

    from ome_zarr.io import parse_url
    from ome_zarr.writer import write_image, write_plate_metadata, write_well_metadata

    path = "test_ngff_plate.zarr"
    row_names = ["A", "B"]
    col_names = ["1", "2", "3"]
    well_paths = ["A/2", "B/3"]
    field_paths = ["0", "1", "2"]

    # generate data
    mean_val=10
    num_wells = len(well_paths)
    num_fields = len(field_paths)
    size_xy = 128
    size_z = 10
    rng = np.random.default_rng(0)
    data = rng.poisson(mean_val, size=(num_wells, num_fields, size_z, size_xy, size_xy)).astype(np.uint8)

    # write the plate of images and corresponding metadata
    store = parse_url(path, mode="w").store
    root = zarr.group(store=store)
    write_plate_metadata(root, row_names, col_names, well_paths)
    for wi, wp in enumerate(well_paths):
        row, col = wp.split("/")
        row_group = root.require_group(row)
        well_group = row_group.require_group(col)
        write_well_metadata(well_group, field_paths)
        for fi, field in enumerate(field_paths):
            image_group = well_group.require_group(str(field))
            write_image(image=data[wi, fi], group=image_group, axes="zyx", storage_options=dict(chunks=(1, size_xy, size_xy)))


This image can be viewed in `napari` using the
`napari-ome-zarr <https://github.com/ome/napari-ome-zarr>`_ plugin::

    import napari

    viewer = napari.Viewer()
    viewer.open(path, plugin="napari-ome-zarr")


Reading OME-NGFF images
-----------------------

This sample code reads an image stored on remote s3 server, but the same
code can be used to read data on a local file system. In either case,
the data is available as `dask` arrays::

    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader
    import napari

    url = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr"

    # read the image data
    store = parse_url(url, mode="r").store

    reader = Reader(parse_url(url))
    # nodes may include images, labels etc
    nodes = list(reader())
    # first node will be the image pixel data
    image_node = nodes[0]

    dask_data = image_node.data

    # We can view this in napari
    # NB: image axes are CZYX: split channels by C axis=0
    viewer = napari.view_image(dask_data, channel_axis=0)
    if __name__ == '__main__':
        napari.run()
