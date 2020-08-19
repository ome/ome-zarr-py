"""
This module is a napari plugin.

It implements the ``napari_get_reader`` hook specification, (to create
a reader plugin).
"""


try:
    from napari_plugin_engine import napari_hook_implementation
except ImportError:

    def napari_hook_implementation(func, *args, **kwargs):
        return func


import logging

# for optional type hints only, otherwise you can delete/ignore this stuff
from typing import List, Optional, Union, Any, Tuple, Dict, Callable

from .utils import parse_url

LOGGER = logging.getLogger("ome_zarr.napari")

LayerData = Union[Tuple[Any], Tuple[Any, Dict], Tuple[Any, Dict, str]]
PathLike = Union[str, List[str]]
ReaderFunction = Callable[[PathLike], List[LayerData]]
# END type hint stuff.


@napari_hook_implementation
def napari_get_reader(path: PathLike) -> Optional[ReaderFunction]:
    """
    Returns a reader for supported paths that include IDR ID

    - URL of the form: https://s3.embassy.ebi.ac.uk/idr/zarr/v0.1/ID.zarr/
    """
    if isinstance(path, list):
        path = path[0]
    instance = parse_url(path)
    if instance is not None and instance.is_zarr():
        return instance.get_reader_function()
    # Ignoring this path
    return None
