"""
Utility methods for ome_zarr access
"""

import json
import logging
import os
from typing import Iterator, List

import dask.array as da
import zarr
from dask.diagnostics import ProgressBar

from .io import parse_url
from .reader import Layer, Multiscales, Reader
from .types import JSONDict

LOGGER = logging.getLogger("ome_zarr.utils")


def info(path: str) -> Iterator[Layer]:
    """
    print information about the ome-zarr fileset
    """
    zarr = parse_url(path)
    assert zarr, f"not a zarr: {zarr}"
    reader = Reader(zarr)
    for layer in reader():

        if not layer.specs:
            print(f"not an ome-zarr: {zarr}")
            continue

        print(layer)
        print(" - metadata")
        for spec in layer.specs:
            print(f"   - {spec.__class__.__name__}")
        print(" - data")
        for array in layer.data:
            print(f"   - {array.shape}")
        LOGGER.debug(layer.data)
        yield layer


def download(input_path: str, output_dir: str = ".") -> None:
    """
    download zarr from URL
    """

    location = parse_url(input_path)
    assert location, f"not a zarr: {location}"

    reader = Reader(location)
    layers: List[Layer] = list()
    paths: List[str] = list()
    for layer in reader():
        layers.append(layer)
        paths.append(layer.zarr.zarr_path)

    strip_common_prefix(paths)

    assert not os.path.exists(output_dir), f"{output_dir} already exists!"
    print("downloading...")
    for path in paths:
        print("  ", path)
    print(f"to {output_dir}")

    for path, layer in sorted(zip(paths, layers)):
        target_dir = os.path.join(output_dir, f"{path}")
        resolutions: List[da.core.Array] = []
        datasets: List[str] = []
        for spec in layer.specs:
            if isinstance(spec, Multiscales):
                datasets = spec.datasets
                resolutions = layer.data
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
            f.write(json.dumps(layer.zarr.zgroup))
        with open(os.path.join(target_dir, ".zattrs"), "w") as f:
            metadata: JSONDict = {}
            layer.write_metadata(metadata)
            f.write(json.dumps(metadata))


def strip_common_prefix(paths: List[str]) -> None:
    parts: List[List[str]] = [x.split(os.path.sep) for x in paths]

    first_mismatch = 0
    min_length = min([len(x) for x in parts])

    for idx in range(min_length):
        if len(set([x[idx] for x in parts])) == 1:
            first_mismatch += 1
        else:
            break

    if first_mismatch <= 0:
        msg = "No common prefix:\n"
        for path in parts:
            msg += f"{path}\n"
        raise Exception(msg)

    for idx, path in enumerate(parts):
        base = os.path.sep.join(path[first_mismatch - 1 :])
        paths[idx] = base
