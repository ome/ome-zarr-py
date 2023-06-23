"""Utility methods for ome_zarr access."""

import json
import logging
import os
import webbrowser
from http.server import (  # type: ignore[attr-defined]
    HTTPServer,
    SimpleHTTPRequestHandler,
    test,
)
from pathlib import Path
from typing import Iterator, List

import dask
import dask.array as da
import zarr
from dask.diagnostics import ProgressBar

from .io import parse_url
from .reader import Multiscales, Node, Reader
from .types import JSONDict

LOGGER = logging.getLogger("ome_zarr.utils")


def info(path: str, stats: bool = False) -> Iterator[Node]:
    """Print information about an OME-Zarr fileset.

    All :class:`Nodes <ome_utils.reader.Node>` that are found from the given path will
    be visited recursively.
    """
    zarr = parse_url(path)
    assert zarr, f"not a zarr: {zarr}"
    reader = Reader(zarr)
    for node in reader():
        if not node.specs:
            print(f"not an ome-zarr node: {node}")
            continue

        print(node)
        print(" - metadata")
        for spec in node.specs:
            print(f"   - {spec.__class__.__name__}")
        print(" - data")
        for array in node.data:
            minmax = ""
            if stats:
                minmax = f" minmax={dask.compute(array.min(), array.max())}"
            print(f"   - {array.shape}{minmax}")
        LOGGER.debug(node.data)
        yield node


def view(input_path: str, port: int = 8000) -> None:
    # serve the parent directory in a simple server with CORS. Open browser

    parent_dir, image_name = os.path.split(input_path)
    parent_dir = str(parent_dir)

    class CORSRequestHandler(SimpleHTTPRequestHandler):
        def end_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            SimpleHTTPRequestHandler.end_headers(self)

        def translate_path(self, path: str) -> str:
            # Since we don't call the class constructor ourselves,
            # we set the directory here instead
            self.directory = parent_dir
            super_path = super().translate_path(path)
            return super_path

    # open ome-ngff-validator in a web browser...
    url = (
        f"https://ome.github.io/ome-ngff-validator/"
        f"?source=http://localhost:{port}/{image_name}"
    )
    webbrowser.open(url)

    # ...then start serving content
    test(CORSRequestHandler, HTTPServer, port=port)


def download(input_path: str, output_dir: str = ".") -> None:
    """Download an OME-Zarr from the given path.

    All :class:`Nodes <ome_utils.reader.Node>` that are found from the given path will
    be included in the download.
    """
    location = parse_url(input_path)
    assert location, f"not a zarr: {location}"

    reader = Reader(location)
    nodes: List[Node] = list()
    paths: List[List[str]] = list()
    for node in reader():
        nodes.append(node)
        paths.append(node.zarr.parts())

    common = strip_common_prefix(paths)
    output_path = Path(output_dir)
    root_path = output_path / common

    assert not root_path.exists(), f"{root_path} already exists!"
    print("downloading...")
    for path in paths:
        print("  ", Path(*path))
    print(f"to {output_dir}")

    for path, node in sorted(zip(paths, nodes)):
        target_path = output_path / Path(*path)
        target_path.mkdir(parents=True)

        with (target_path / ".zgroup").open("w") as f:
            f.write(json.dumps(node.zarr.zgroup))
        with (target_path / ".zattrs").open("w") as f:
            metadata: JSONDict = {}
            node.write_metadata(metadata)
            f.write(json.dumps(metadata))

        resolutions: List[da.core.Array] = []
        datasets: List[str] = []
        for spec in node.specs:
            if isinstance(spec, Multiscales):
                datasets = spec.datasets
                resolutions = node.data
                if datasets and resolutions:
                    pbar = ProgressBar()
                    for dataset, data in reversed(list(zip(datasets, resolutions))):
                        LOGGER.info("resolution %s...", dataset)
                        with pbar:
                            data.to_zarr(str(target_path / dataset))
            else:
                # Assume a group that needs metadata, like labels
                zarr.group(str(target_path))


def strip_common_prefix(parts: List[List[str]]) -> str:
    """Find and remove the prefix common to all strings.

    Returns the last element of the common prefix.
    An exception is thrown if no common prefix exists.

    >>> paths = [["a", "b"], ["a", "b", "c"]]
    >>> strip_common_prefix(paths)
    'b'
    >>> paths
    [['b'], ['b', 'c']]
    """
    first_mismatch = 0
    min_length = min(len(x) for x in parts)

    for idx in range(min_length):
        if len({x[idx] for x in parts}) == 1:
            first_mismatch += 1
        else:
            break

    if first_mismatch <= 0:
        msg = "No common prefix:\n"
        for path in parts:
            msg += f"{path}\n"
        raise Exception(msg)
    else:
        common = parts[0][first_mismatch - 1]

    for idx, path in enumerate(parts):
        parts[idx] = parts[idx][first_mismatch - 1 :]

    return common
