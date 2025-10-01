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

The default version of OME-NGFF is v0.5, is based on Zarr v3. A zarr v3 group and store is created
by `zarr.open_group()` below. To write OME-NGFF v0.4 (Zarr v2), add the `zarr_format=2` argument.

The following code creates a 3D Image in OME-Zarr::

    import numpy as np
    import zarr

    from ome_zarr.format import FormatV04
    from ome_zarr.writer import write_image, add_metadata

    path = "test_ngff_image.zarr"

    size_xy = 128
    size_z = 10
    rng = np.random.default_rng(0)
    data = rng.poisson(lam=10, size=(size_z, size_xy, size_xy)).astype(np.uint8)

    # Add fmt=FormatV04() parameter to write v0.4 format (zarr v2)
    write_image(data, path, axes="zyx")


This image can be viewed in `napari` using the
`napari-ome-zarr <https://github.com/ome/napari-ome-zarr>`_ plugin::

    $ napari test_ngff_image.zarr

Rendering settings
------------------
Rendering settings can be added to an existing zarr group::

    add_metadata(path, {"omero": {
        "channels": [{
            "color": "00FFFF",
            "window": {"start": 0, "end": 20, "min": 0, "max": 255},
            "label": "random",
            "active": True,
        }]
    }})

Writing labels
--------------

The following code creates a 3D Image in OME-Zarr with labels::

    import numpy as np
    import zarr
    import os

    from skimage.data import binary_blobs
    from ome_zarr.format import FormatV04
    from ome_zarr.writer import write_image, add_metadata

    path = "test_ngff_image_labels.zarr"
    os.mkdir(path)

    mean_val=10
    size_xy = 128
    size_z = 10
    rng = np.random.default_rng(0)
    data = rng.poisson(mean_val, size=(size_z, size_xy, size_xy)).astype(np.uint8)

    # Use zarr_format=2 to write v0.4 format (zarr v2)
    root = zarr.open_group(path, mode="w")
    write_image(image=data, group=root, axes="zyx",
                storage_options=dict(chunks=(1, size_xy, size_xy)))
    # optional rendering settings
    add_metadata(root, {"omero": {
        "channels": [{
            "color": "00FFFF",
            "window": {"start": 0, "end": 20, "min": 0, "max": 255},
            "label": "random",
            "active": True,
        }]
    }})


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
    add_metadata(labels_grp, {"labels": [label_name]})
    label_grp = labels_grp.create_group(label_name)
    write_image(label, label_grp, axes="zyx")

    # we need 'image-label' attr to be recognized as label
    add_metadata(label_grp, {"image-label": {
        "colors": [
            {"label-value": 1, "rgba": [255, 0, 0, 255]},
            {"label-value": 2, "rgba": [0, 255, 0, 255]},
            {"label-value": 3, "rgba": [255, 255, 0, 255]}
        ]
    }})


Writing HCS datasets to OME-NGFF
--------------------------------

This sample code shows how to write a high-content screening dataset (i.e. culture plate with multiple wells) to a OME-NGFF file::

    import numpy as np
    import zarr

    from ome_zarr.format import FormatV04
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
    # Use zarr_format=2 to write v0.4 format (zarr v2)
    root = zarr.open_group(path, mode="w")
    write_plate_metadata(root, row_names, col_names, well_paths)
    for wi, wp in enumerate(well_paths):
        row, col = wp.split("/")
        row_group = root.require_group(row)
        well_group = row_group.require_group(col)
        write_well_metadata(well_group, field_paths)
        for fi, field in enumerate(field_paths):
            image_group = well_group.require_group(str(field))
            write_image(image=data[wi, fi], group=image_group, axes="zyx",
                        storage_options=dict(chunks=(1, size_xy, size_xy)))


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

    from dask import array as da
    import zarr
    import napari

    url = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr"

    # read the image data
    root = zarr.open_group(url)
    zattrs = root.attrs.asdict()
    # Handle v0.5+ - unwrap 'ome' namespace
    if "ome" in zattrs:
        zattrs = zattrs["ome"]

    paths = [ds["path"] for ds in zattrs["multiscales"][0]["datasets"]]
    dask_data = [da.from_zarr(root[path]) for path in paths]

    # We can view this in napari
    # NB: image axes are CZYX: split channels by C axis=0
    viewer = napari.view_image(dask_data, channel_axis=0)
    if __name__ == '__main__':
        napari.run()


More writing examples
---------------------

Writing big image from tiles::

    # Created for https://forum.image.sc/t/writing-tile-wise-ome-zarr-with-pyramid-size/85063

    import os
    import zarr
    from ome_zarr.format import CurrentFormat, FormatV04
    from ome_zarr.reader import Reader
    from ome_zarr.writer import write_multiscales_metadata
    from ome_zarr.dask_utils import resize as da_resize
    import numpy as np
    import dask.array as da
    from math import ceil

    fmt = CurrentFormat()
    # Use fmt=FormatV04() to write v0.4 format (zarr v2)

    url = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.3/9836842.zarr"
    root = zarr.open_group(url)
    zattrs = root.attrs.asdict()
    # Handle v0.5+ - unwrap 'ome' namespace
    if "ome" in zattrs:
        zattrs = zattrs["ome"]
    paths = [ds["path"] for ds in zattrs["multiscales"][0]["datasets"]]
    # first level of the pyramid
    dask_data = da.from_zarr(root[paths[0]])
    tile_size = 512
    axes = [{"name": "c", "type": "channel"}, {"name": "y", "type": "space"}, {"name": "x", "type": "space"}]

    def downsample_pyramid_on_disk(parent, paths):
        """
        Takes a high-resolution Zarr array at paths[0] in the zarr group
        and down-samples it by a factor of 2 for each of the other paths
        """
        group_path = str(parent.store_path)
        img_path = parent.store_path / parent.path
        image_path = os.path.join(group_path, parent.path)
        print("downsample_pyramid_on_disk", image_path)
        for count, path in enumerate(paths[1:]):
            target_path = os.path.join(image_path, path)
            if os.path.exists(target_path):
                print("path exists: %s" % target_path)
                continue
            # open previous resolution from disk via dask...
            path_to_array = os.path.join(image_path, paths[count])
            dask_image = da.from_zarr(path_to_array)

            # resize in X and Y
            dims = list(dask_image.shape)
            dims[-1] = dims[-1] // 2
            dims[-2] = dims[-2] // 2
            output = da_resize(
                dask_image, tuple(dims), preserve_range=True, anti_aliasing=False
            )

            options = {}
            if fmt.zarr_format == 2:
                options["dimension_separator"] = "/"
            else:
                options["chunk_key_encoding"] = fmt.chunk_key_encoding
                options["dimension_names"] = [axis["name"] for axis in axes]
            # write to disk
            da.to_zarr(
                arr=output, url=img_path, component=path,
                zarr_format=fmt.zarr_format, **options
            )
        return paths

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

    root = zarr.open_group("9836842.zarr", mode="w")

    # create empty array at root of pyramid
    zarray = root.require_array(
        "0",
        shape=shape,
        exact=True,
        chunks=chunks,
        dtype=d_type,
        chunk_key_encoding=fmt.chunk_key_encoding,
        dimension_names=[axis["name"] for axis in axes], # omit for v0.4
    )

    print("row_count", row_count, "col_count", col_count)
    # Go through all tiles and write data to "0" array
    for ch_index in range(channel_count):
        for row in range(row_count):
            for col in range(col_count):
                tile = get_tile(ch_index, row, col).compute()
                y1 = row * tile_size
                y2 = y1 + tile_size
                x1 = col * tile_size
                x2 = x1 + tile_size
                print("ch_index", ch_index, "row", row, "col", col)
                zarray[ch_index, y1:y2, x1:x2] = tile

    paths = ["0", "1", "2"]

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


Using dask to fetch. Here concatenate lazy "delayed" source of tiles into a full image.
When that dask data is passed to write_image() the tiles will be loaded on the fly::

    # Created for https://forum.image.sc/t/writing-tile-wise-ome-zarr-with-pyramid-size/85063

    import dask.array as da
    import numpy as np
    import zarr
    from dask import delayed

    from ome_zarr.format import FormatV04
    from ome_zarr.writer import write_image, add_metadata

    zarr_name = "test_dask.zarr"
    root = zarr.open_group(zarr_name, mode="w")

    size_xy = 100
    channel_count = 2
    size_z = 10
    row_count = 3
    col_count = 5
    dtype = np.uint8
    tile_shape = (size_xy, size_xy)


    def get_tile(ch, z, row, column):
        print("get_tile", ch, z, row, column)
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

    # This will create a downsampled 'multiscales' pyramid
    write_image(dask_data, root, axes="czyx")

    add_metadata(root, {"omero": {
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
    }})

    print("Created image. Open with...")
    print(f"ome_zarr view {zarr_name}")
