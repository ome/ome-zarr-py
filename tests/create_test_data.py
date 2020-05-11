#!/usr/bin/env python
import argparse
import zarr
import numpy as np
import os
import json
from skimage import data
from skimage.transform import pyramid_gaussian, pyramid_laplacian


def create_zarr(zarr_directory):

    base = np.tile(data.astronaut(), (2, 2, 1))
    gaussian = list(
        pyramid_gaussian(base, downscale=2, max_layer=4, multichannel=True)
    )

    pyramid = []
    # convert each level of pyramid into 5D image (t, c, z, y, x)
    for pixels in gaussian:
        red = pixels[:,:,0]
        green = pixels[:,:,1]
        blue = pixels[:,:,2]
        # wrap to make 5D: (t, c, z, y, x)
        pixels = np.array([np.array([red]), np.array([green]), np.array([blue])])
        pixels = np.array([pixels])
        pyramid.append(pixels)

    store = zarr.DirectoryStore(zarr_directory)
    grp = zarr.group(store)
    paths = []
    for path, dataset in enumerate(pyramid):
        grp.create_dataset(str(path), data=pyramid[path])
        paths.append({"path": str(path)})

    image_data = {
        'id': 1,
        'channels': [{
            'color': 'FF0000',
            'window': {'start': 0, 'end': 1},
            'label': 'Red',
            'active': True,
        },{
            'color': '00FF00',
            'window': {'start': 0, 'end': 1},
            'label': 'Green',
            'active': True,
        },{
            'color': '0000FF',
            'window': {'start': 0, 'end': 1},
            'label': 'Blue',
            'active': True,
        }],
        'rdefs': {
            'model': 'color',
        }
    }

    multiscales = [{
        "version": "0.1",
        "datasets": paths,
    }]
    grp.attrs["multiscales"] = multiscales
    grp.attrs["omero"] = image_data
