"""This module is a napari plugin.

It implements the ``napari_get_reader`` hook specification, (to create a reader plugin).
"""


import logging
import warnings
from typing import Any, Callable, Dict, Iterator, List, Optional

from .data import CHANNEL_DIMENSION
from .io import parse_url
from .reader import Label, Node, Reader
from .types import LayerData, PathLike, ReaderFunction

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
    """Returns a reader for supported paths that include IDR ID.

    - URL of the form: https://s3.embassy.ebi.ac.uk/idr/zarr/v0.1/ID.zarr/
    """
    if isinstance(path, list):
        if len(path) > 1:
            warnings.warn("more than one path is not currently supported")
        path = path[0]
    zarr = parse_url(path)
    if zarr:
        reader = Reader(zarr)
        return transform(reader())
    # Ignoring this path
    return None


def transform(nodes: Iterator[Node]) -> Optional[ReaderFunction]:
    def f(*args: Any, **kwargs: Any) -> List[LayerData]:
        results: List[LayerData] = list()

        for node in nodes:
            data: List[Any] = node.data
            metadata: Dict[str, Any] = node.metadata
            if data is None or len(data) < 1:
                LOGGER.debug(f"skipping non-data {node}")
            else:
                LOGGER.debug(f"transforming {node}")
                shape = data[0].shape

                layer_type: str = "image"
                if node.load(Label):
                    layer_type = "labels"
                    if "colormap" in metadata:
                        del metadata["colormap"]

                elif shape[CHANNEL_DIMENSION] > 1:
                    metadata["channel_axis"] = CHANNEL_DIMENSION
                else:
                    for x in ("name", "visible", "contrast_limits", "colormap"):
                        if x in metadata:
                            try:
                                metadata[x] = metadata[x][0]
                            except Exception:
                                del metadata[x]

                rv: LayerData = (data, metadata, layer_type)
                LOGGER.debug(f"Transformed: {rv}")
                results.append(rv)

        return results

    return f
