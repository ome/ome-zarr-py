"""This module is a napari plugin.

It implements the ``napari_get_reader`` hook specification, (to create a reader plugin).
"""


import logging
import warnings
from typing import Any, Callable, Dict, Iterator, List, Optional, Set

from .data import CHANNEL_DIMENSION
from .io import parse_url
from .reader import Label, Node, PlateLabels, Reader
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
                if node.load(Label) or node.load(PlateLabels):
                    layer_type = "labels"
                    if "colormap" in metadata:
                        del metadata["colormap"]
                    if "properties" in metadata:
                        props = metadata["properties"]
                        reader_props = {}
                        label_indices = list(props.keys())
                        reader_props["index"] = label_indices

                        # properties may be ragged, so we need all possible properties
                        all_keys: Set[str]
                        all_keys = set()
                        for index in label_indices:
                            all_keys = all_keys.union(set(props[index].keys()))

                        # napari expects lists of equal length so we must fill with None
                        for prop_key in all_keys:
                            reader_props[prop_key] = [
                                props[i][prop_key] if prop_key in props[i] else None
                                for i in label_indices
                            ]
                        metadata["properties"] = reader_props

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

                # the 'metadata' dict takes any extra info, not supported
                # by napari. If we have 'plate' info, add a shapes layer
                if "metadata" in metadata:
                    if "plate" in metadata["metadata"]:
                        plate_info = metadata["metadata"]["plate"]
                        plate_width = shape[-1]
                        plate_height = shape[-2]
                        rows = len(plate_info["rows"])
                        columns = len(plate_info["columns"])
                        well_width = plate_width / columns
                        well_height = plate_height / rows
                        labels = []
                        outlines = []
                        for row in range(rows):
                            for column in range(columns):
                                x1 = int(column * well_width)
                                x2 = int((column + 1) * well_width)
                                y1 = int(row * well_height)
                                y2 = int((row + 1) * well_height)
                                # Since napari will only place labels 'outside' a
                                # bounding box we have a line
                                # along top of Well, with label below
                                outlines.append([[y1, x1], [y1, x2]])
                                row_name = plate_info["rows"][row]["name"]
                                col_name = plate_info["columns"][column]["name"]
                                label = f"{row_name}{col_name}"
                                labels.append(label)
                                # Well bounding box, with no label
                                outlines.append(
                                    [[y1, x1], [y2, x1], [y2, x2], [y1, x2]]
                                )
                                labels.append("")
                        text_parameters = {
                            "text": "{well}",
                            "size": 12,
                            "color": "white",
                            "anchor": "lower_left",
                            "translation": [5, 5],
                        }
                        shapes: LayerData = (
                            outlines,
                            {
                                "edge_color": "white",
                                "face_color": "transparent",
                                "name": "Well Labels",
                                "text": text_parameters,
                                "properties": {"well": labels},
                            },
                            "shapes",
                        )
                        results.append(shapes)

        return results

    return f
