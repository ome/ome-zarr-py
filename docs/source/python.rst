ome_zarr python library
=======================

Writing OME-NGFF images
-----------------------

The principle entry-point for writing OME-NGFF images is :py:func:`ome_zarr.writer.write_image`.
This takes an n-dimensional `numpy` array and writes it to the specified `zarr group` according
to the OME-NGFF specification.
By default, a pyramid of resolution levels will be created by down-sampling the data by a factor
of 2 in the X and Y dimensions.

Alternatively, the :py:func:`ome_zarr.writer.write_multiscale` can be used, which takes a
"pyramid" `numpy` arrays.

The following code creates a 3D Image in OME-Zarr with labels::

    import numpy as np
    import zarr
    import os

    from skimage.data import binary_blobs
    from ome_zarr.io import parse_url
    from ome_zarr.writer import write_image

    path = "test_ngff_image"
    os.mkdir(path)

    mean_val=10
    size_xy = 128
    size_z = 10
    rng = np.random.default_rng(0)
    data = rng.poisson(mean_val, size=(size_z, size_xy, size_xy)).astype(np.uint8)

    # write the image data
    store = parse_url(path, mode="w").store
    root = zarr.group(store=store)
    write_image(image=data, group=root, chunks=(1, size_xy, size_xy), axes="zyx")
    # optional rendering settings
    root.attrs["omero"] = {
        "channels": [{
            "color": "00FFFF",
            "window": {"start": 0, "end": 20},
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


This image can be viewed in `napari` using the
`napari-ome-zarr <https://github.com/ome/napari-ome-zarr>`_ plugin::

    $ napari test_ngff_image