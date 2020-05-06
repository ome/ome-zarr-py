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
import re
import zarr
import requests
import dask.array as da
from vispy.color import Colormap

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
    if isinstance(path, str):
        if re.search(r'^.*/(\d+)\.zarr.*', path) is not None:
            return reader_function


def reader_function(path: PathLike) -> List[LayerData]:
    """Take a path or list of paths and return a list of LayerData tuples."""
    if isinstance(path, list):
        path = path[0]
    return [load_omero_zarr(path)]


def load_omero_metadata(zarr_path):
    """Load OMERO metadata as json and convert for napari"""
    omero_path = zarr_path + "omero.json"
    metadata = {}
    try:
        image_data = requests.get(omero_path).json()
        print(image_data)
        colormaps = []
        for ch in image_data['channels']:
            # 'FF0000' -> [1, 0, 0]
            rgb = [(int(ch['color'][i:i+2], 16)/255) for i in range(0, 6, 2)]
            if image_data['rdefs']['model'] == 'greyscale':
                rgb = [1, 1, 1]
            colormaps.append(Colormap([[0, 0, 0], rgb]))
        metadata['colormap'] = colormaps
        metadata['contrast_limits'] = [[ch['window']['start'], ch['window']['end']] for ch in image_data['channels']]
        metadata['name'] = [ch['label'] for ch in image_data['channels']]
        metadata['visible'] = [ch['active'] for ch in image_data['channels']]
    except Exception:
        pass

    return metadata


def load_omero_zarr(path):
    zarr_path = path.endswith("/") and path or f"{path}/"
    attrs_path = zarr_path + ".zattrs"
    root_attrs = requests.get(attrs_path).json()

    resolutions = ["0"]  # TODO: could be first alphanumeric dataset on err
    try:
        print('root_attrs', root_attrs)
        if 'multiscales' in root_attrs:
            datasets = root_attrs['multiscales'][0]['datasets']
            resolutions = [d['path'] for d in datasets]
        print('resolutions', resolutions)
    except Exception as e:
        raise e

    pyramid = []
    for resolution in resolutions:
        # data.shape is (t, c, z, y, x) by convention
        data = da.from_zarr(f"{zarr_path}{resolution}")
        chunk_sizes = [str(c[0]) + (" (+ %s)" % c[-1] if c[-1] != c[0] else '') for c in data.chunks]
        print('resolution', resolution, 'shape (t, c, z, y, x)', data.shape, 'chunks', chunk_sizes, 'dtype', data.dtype)
        pyramid.append(data)

    metadata = load_omero_metadata(zarr_path)

    return(pyramid, {'channel_axis': 1, **metadata})

    # TODO: would be nice to be able to:
    # viewer.dims.set_axis_label(0, 'T')
    # viewer.dims.set_point(0, image_data['rdefs']['defaultT'])
    # viewer.dims.set_axis_label(1, 'Z')
    # viewer.dims.set_point(1, image_data['rdefs']['defaultZ'])
