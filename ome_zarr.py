"""
This module is a napari plugin.

It implements the ``napari_get_reader`` hook specification, (to create
a reader plugin).

Type annotations here are OPTIONAL!
If you don't care to annotate the return types of your functions
your plugin doesn't need to import, or even depend on napari at all!

Replace code below accordingly.
"""
import numpy as np
import s3fs
import os
import re
import json
import zarr
import requests
import dask.array as da
import warnings

from dask.diagnostics import ProgressBar
from vispy.color import Colormap

from urllib.parse import urlparse
from pluggy import HookimplMarker

import logging
# DEBUG logging for s3fs so we can track remote calls
logging.basicConfig(level=logging.INFO)
logging.getLogger('s3fs').setLevel(logging.DEBUG)

# for optional type hints only, otherwise you can delete/ignore this stuff
from typing import List, Optional, Union, Any, Tuple, Dict, Callable

LayerData = Union[Tuple[Any], Tuple[Any, Dict], Tuple[Any, Dict, str]]
PathLike = Union[str, List[str]]
ReaderFunction = Callable[[PathLike], List[LayerData]]
# END type hint stuff.

napari_hook_implementation = HookimplMarker("napari")



@napari_hook_implementation
def napari_get_reader(path: PathLike) -> Optional[ReaderFunction]:
    """
    Returns a reader for supported paths that include IDR ID

    - URL of the form: https://s3.embassy.ebi.ac.uk/idr/zarr/v0.1/ID.zarr/
    """
    if isinstance(path, list):
        path = path[0]
    instance = parse_url(path)
    if instance.is_zarr():
        return instance.get_reader_function()


def parse_url(path):
    result = urlparse(path)
    if result.scheme in ("", "file"):
        # Strips 'file://' if necessary
        return LocalZarr(result.path)
    else:
        return RemoteZarr(path)


class BaseZarr:

    def __init__(self, path):
        self.zarr_path = path.endswith("/") and path or f"{path}/"
        self.zarray = self.get_json(".zarray")
        self.zgroup = self.get_json(".zgroup")
        if self.zgroup:
            self.root_attrs = self.get_json(".zattrs")
            if "omero" in self.root_attrs:
                self.image_data = self.root_attrs["omero"]
                # TODO: start checking metadata version
            else:
                # Backup location that can be removed in the future.
                warnings.warn("deprecated loading of omero.josn",
                              DeprecationWarning)
                self.image_data = self.get_json("omero.json")

    def __str__(self):
        suffix = ""
        if self.zgroup:
            suffix += " [zgroup]"
        if self.zarray:
            suffix += " [zarray]"
        return f"{self.zarr_path}{suffix}"

    def is_zarr(self):
        return self.zarray or self.zgroup

    def is_ome_zarr(self):
        return self.zgroup and "multiscales" in self.root_attrs

    def get_json(self, subpath):
        raise NotImplementedError("unknown")

    def get_reader_function(self):
        if not self.is_zarr():
            raise Exception(f"not a zarr: {self}")
        return self.reader_function

    def reader_function(self, path: PathLike) -> List[LayerData]:
        """Take a path or list of paths and return a list of LayerData tuples."""

        if isinstance(path, list):
            path = path[0]
            # TODO: safe to ignore this path?

        if self.is_ome_zarr():
            return [self.load_ome_zarr()]

        elif self.zarray:
            data = da.from_zarr(f"{self.zarr_path}")
            return [(data, {'channel_axis': 1})]

    def load_omero_metadata(self):
        """Load OMERO metadata as json and convert for napari"""
        metadata = {}
        try:
            channels = self.image_data['channels']
            colormaps = []
            for ch in channels:
                # 'FF0000' -> [1, 0, 0]
                rgb = [(int(ch['color'][i:i+2], 16)/255) for i in range(0, 6, 2)]
                if self.image_data['rdefs']['model'] == 'greyscale':
                    rgb = [1, 1, 1]
                colormaps.append(Colormap([[0, 0, 0], rgb]))
            metadata['colormap'] = colormaps
            metadata['contrast_limits'] = [[ch['window']['start'], ch['window']['end']] for ch in channels]
            metadata['name'] = [ch['label'] for ch in channels]
            metadata['visible'] = [ch['active'] for ch in channels]
        except Exception as e:
            print(e)

        return metadata


    def load_ome_zarr(self):

        resolutions = ["0"]  # TODO: could be first alphanumeric dataset on err
        try:
            print('root_attrs', self.root_attrs)
            if 'multiscales' in self.root_attrs:
                datasets = self.root_attrs['multiscales'][0]['datasets']
                resolutions = [d['path'] for d in datasets]
            print('resolutions', resolutions)
        except Exception as e:
            raise e

        pyramid = []
        for resolution in resolutions:
            # data.shape is (t, c, z, y, x) by convention
            data = da.from_zarr(f"{self.zarr_path}{resolution}")
            chunk_sizes = [str(c[0]) + (" (+ %s)" % c[-1] if c[-1] != c[0] else '') for c in data.chunks]
            print('resolution', resolution, 'shape (t, c, z, y, x)', data.shape, 'chunks', chunk_sizes, 'dtype', data.dtype)
            pyramid.append(data)

        if len(pyramid) == 1:
            pyramid = pyramid[0]
        metadata = self.load_omero_metadata()
        return (pyramid, {'channel_axis': 1, **metadata})



class LocalZarr(BaseZarr):

    def get_json(self, subpath):
        filename = os.path.join(self.zarr_path, subpath)

        if not os.path.exists(filename):
            return {}

        with open(filename) as f:
            return json.loads(f.read())


class RemoteZarr(BaseZarr):

    def get_json(self, subpath):
        rsp = requests.get(f"{self.zarr_path}{subpath}")
        try:
            if rsp.status_code in (403, 404):  # file doesn't exist
                return {}
            return rsp.json()
        except:
            print("FIXME", rsp.status_code, rsp.text)
            return {}


def info(path):
    """
    print information about the ome-zarr fileset
    """
    zarr = parse_url(path)
    if not zarr.is_ome_zarr():
        print(f"not an ome-zarr: {zarr}")
        return
    reader = zarr.get_reader_function()
    data = reader(path)
    print(data)


def download(path, output_dir='.', zarr_name=''):
    """
    download zarr from URL
    """
    omezarr = parse_url(path)
    if not omezarr.is_ome_zarr():
        print(f"not an ome-zarr: {path}")

    image_id = omezarr.image_data.get('id', 'unknown')
    print('image_id', image_id)
    if not zarr_name:
        zarr_name = f'{image_id}.zarr'

    try:
        datasets = [x['path'] for x in omezarr.root_attrs["multiscales"][0]["datasets"]]
    except KeyError:
        datasets = ["0"]
    print('datasets', datasets)
    resolutions = [da.from_zarr(path, component=str(i)) for i in datasets]
    # levels = list(range(len(resolutions)))

    target_dir = os.path.join(output_dir, f'{zarr_name}')
    print(f'downloading to {target_dir}')

    pbar = ProgressBar()
    for dataset, data in reversed(list(zip(datasets, resolutions))):
        print(f'resolution {dataset}...')
        with pbar:
            data.to_zarr(os.path.join(target_dir, dataset))

    with open(os.path.join(target_dir, '.zgroup'), 'w') as f:
        f.write(json.dumps(omezarr.zgroup))
    with open(os.path.join(target_dir, '.zattrs'), 'w') as f:
        f.write(json.dumps(omezarr.root_attrs))
