"""Utility methods for ome_zarr access."""

import json
import logging
from pathlib import Path
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
            print(f"not an ome-zarr node: {node}")
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
                        LOGGER.info(f"resolution {dataset}...")
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
