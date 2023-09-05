ome_zarr python library
=======================

Writing OME-NGFF images
-----------------------

The principle entry-point for writing OME-NGFF images is :py:func:`ome_zarr.writer.write_image`.
This takes an n-dimensional `numpy` array or `dask` array and writes it to the specified `zarr group` according
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

    $ napari test_ngff_image.zarr


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


More writing examples
---------------------

Writing big image from tiles:

    # Created for https://forum.image.sc/t/writing-tile-wise-ome-zarr-with-pyramid-size/85063

    import zarr
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader
    from ome_zarr.writer import write_multiscales_metadata
    from omero_zarr.raw_pixels import downsample_pyramid_on_disk
    import numpy as np
    from math import ceil

    url = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.3/9836842.zarr"
    reader = Reader(parse_url(url))
    nodes = list(reader())
    # first level of the pyramid
    dask_data = nodes[0].data[0]
    tile_size = 512

    def get_tile(ch, row, col):
        # read the tile data from somewhere - we use the dask array
        y1 = row * tile_size
        y2 = y1 + tile_size
        x1 = col * tile_size
        x2 = x1 + tile_size
        return dask_data[ch, y1:y2, x1:x2]

    # (4,1920,1920)
    shape = dask_data.shape
    chunks = (1, tile_size, tile_size)
    d_type = np.dtype('<u2')

    channel_count = shape[0]
    row_count = ceil(shape[-2]/tile_size)
    col_count = ceil(shape[-1]/tile_size)

    store = parse_url("9836842.zarr", mode="w").store
    root = zarr.group(store=store)

    # create empty array at root of pyramid
    zarray = root.require_dataset(
        "0",
        shape=shape,
        exact=True,
        chunks=chunks,
        dtype=d_type,
    )

    print("row_count", row_count, "col_count", col_count)
    # Go through all tiles and write data to "0" array
    for ch_index in range(channel_count):
        for row in range(row_count):
            for col in range(col_count):
                tile = get_tile(ch_index, row, col)
                y1 = row * tile_size
                y2 = y1 + tile_size
                x1 = col * tile_size
                x2 = x1 + tile_size
                zarray[ch_index, y1:y2, x1:x2] = tile

    paths = ["0", "1", "2"]
    axes = [{"name": "c", "type": "channel"}, {"name": "y", "type": "space"}, {"name": "x", "type": "space"}]

    # We have "0" array. This downsamples (in X and Y dims only) to create "1" and "2"
    downsample_pyramid_on_disk(root, paths)

    transformations = [
        [{"type": "scale", "scale": [1.0, 1.0, 1.0]}],
        [{"type": "scale", "scale": [1.0, 2.0, 2.0]}],
        [{"type": "scale", "scale": [1.0, 4.0, 4.0]}]
    ]
    datasets = []
    for p, t in zip(paths, transformations):
        datasets.append({"path": p, "coordinateTransformations": t})

    write_multiscales_metadata(root, datasets, axes=axes)

Using dask to fetch:

    # Created for https://forum.image.sc/t/writing-tile-wise-ome-zarr-with-pyramid-size/85063

    import dask.array as da
    import numpy as np
    import zarr
    from dask import delayed

    from ome_zarr.io import parse_url
    from ome_zarr.writer import write_image, write_multiscales_metadata

    zarr_name = "test_dask.zarr"
    store = parse_url(zarr_name, mode="w").store
    root = zarr.group(store=store)

    size_xy = 100
    channel_count = 2
    size_z = 10
    row_count = 3
    col_count = 5
    dtype = np.uint8
    tile_shape = (size_xy, size_xy)


    def get_tile(ch, z, row, column):
        mean_val = ((row + 1) * (column + 1) * 4) + (10 * z)
        rng = np.random.default_rng(1000 * ch)
        return rng.poisson(mean_val, size=tile_shape).astype(dtype)


    delayed_reader = delayed(get_tile)

    dask_channels = []

    for ch in range(channel_count):
        dask_planes = []
        for z_index in range(size_z):
            dask_rows = []
            for row in range(row_count):
                dask_tiles = []
                for col in range(col_count):
                    dask_tile = da.from_delayed(
                        delayed_reader(ch, z_index, row, col), shape=tile_shape, dtype=dtype
                    )
                    dask_tiles.append(dask_tile)
                dask_row = da.concatenate(dask_tiles, axis=1)
                dask_rows.append(dask_row)
            dask_plane = da.concatenate(dask_rows, axis=0)
            dask_planes.append(dask_plane)
        # stack 2D planes to 3D for each channel
        dask_channels.append(da.stack(dask_planes, axis=0))
    # stack 3D (zyx) data to 4D (czyx)
    dask_data = da.stack(dask_channels, axis=0)

    print("dask_data", dask_data)

    write_image(dask_data, root, axes="czyx")

    root.attrs["omero"] = {
        "channels": [
            {
                "color": "FF0000",
                "window": {"min": 0, "start": 0, "end": 200, "max": 256},
                "label": "random_red",
                "active": True,
            },
            {
                "color": "00FF00",
                "window": {"min": 0, "start": 0, "end": 200, "max": 256},
                "label": "random_green",
                "active": True,
            },
        ]
    }

    print("Created image. Open with...")
    print(f"ome_zarr view {zarr_name}")
