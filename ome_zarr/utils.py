"""Utility methods for ome_zarr access."""

import json
import logging
import os
from typing import Iterator, List

import dask.array as da
import zarr
from dask.diagnostics import ProgressBar

from .io import parse_url
from .reader import Multiscales, Node, Reader
from .types import JSONDict

LOGGER = logging.getLogger("ome_zarr.utils")


def info(path: str) -> Iterator[Node]:
    """Print information about an OME-Zarr fileset.

    All :class:`Nodes <ome_utils.reader.Node>` that are found from the given path will
    be visited recursively.
    """
    zarr = parse_url(path)
    assert zarr, f"not a zarr: {zarr}"
    reader = Reader(zarr)
    for node in reader():

        if not node.specs:
            print(f"not an ome-zarr: {zarr}")
            continue

        print(node)
        print(" - metadata")
        for spec in node.specs:
            print(f"   - {spec.__class__.__name__}")
        print(" - data")
        for array in node.data:
            print(f"   - {array.shape}")
        LOGGER.debug(node.data)
        yield node


def download(input_path: str, output_dir: str = ".") -> None:
    """Download an OME-Zarr from the given path.

    All :class:`Nodes <ome_utils.reader.Node>` that are found from the given path will
    be included in the download.
    """
    location = parse_url(input_path)
    assert location, f"not a zarr: {location}"

    reader = Reader(location)
    nodes: List[Node] = list()
    paths: List[str] = list()
    for node in reader():
        nodes.append(node)
        paths.append(node.zarr.zarr_path)

    common = strip_common_prefix(paths)
    root = os.path.join(output_dir, common)

    assert not os.path.exists(root), f"{root} already exists!"
    print("downloading...")
    for path in paths:
        print("  ", path)
    print(f"to {output_dir}")

    for path, node in sorted(zip(paths, nodes)):
        target_dir = os.path.join(output_dir, f"{path}")
        resolutions: List[da.core.Array] = []
        datasets: List[str] = []
        for spec in node.specs:
            if isinstance(spec, Multiscales):
                datasets = spec.datasets
                resolutions = node.data
                if datasets and resolutions:
                    pbar = ProgressBar()
                    for dataset, data in reversed(list(zip(datasets, resolutions))):
                        LOGGER.info(f"resolution {dataset}...")
                        with pbar:
                            data.to_zarr(os.path.join(target_dir, dataset))
            else:
                # Assume a group that needs metadata, like labels
                zarr.group(target_dir)

        with open(os.path.join(target_dir, ".zgroup"), "w") as f:
            f.write(json.dumps(node.zarr.zgroup))
        with open(os.path.join(target_dir, ".zattrs"), "w") as f:
            metadata: JSONDict = {}
            node.write_metadata(metadata)
            f.write(json.dumps(metadata))


def strip_common_prefix(paths: List[str]) -> str:
    """Find and remove the prefix common to all strings.

    Returns the last element of the common prefix.
    An exception is thrown if no common prefix exists.

    >>> paths = ["a/b", "a/b/c"]
    >>> strip_common_prefix(paths)
    'b'
    >>> paths
    ['b', 'b/c']
    """
    parts: List[List[str]] = [x.split(os.path.sep) for x in paths]

    first_mismatch = 0
    min_length = min([len(x) for x in parts])

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
        base = os.path.sep.join(path[first_mismatch - 1 :])
        paths[idx] = base

    return common
