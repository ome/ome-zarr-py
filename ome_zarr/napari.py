"""
This module is a napari plugin.

It implements the ``napari_get_reader`` hook specification, (to create
a reader plugin).
"""


import logging

from typing import Any, Callable, Optional
from .reader import PathLike, ReaderFunction
from .utils import parse_url


try:
    from napari_plugin_engine import napari_hook_implementation
except ImportError:

    def napari_hook_implementation(
        func: Callable, *args: Any, **kwargs: Any
    ) -> Callable:
        return func


LOGGER = logging.getLogger("ome_zarr.napari")


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
