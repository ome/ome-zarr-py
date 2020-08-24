"""
Utility methods for ome_zarr access
"""

import json
import logging
import os
from typing import List, Optional

import dask.array as da
from dask.diagnostics import ProgressBar

from .io import parse_url
from .reader import OMERO, Layer, Multiscales
from .types import JSONDict

LOGGER = logging.getLogger("ome_zarr.utils")


def info(path: str) -> Optional[Layer]:
    """
    print information about the ome-zarr fileset
    """
    zarr = parse_url(path)
    if not zarr:
        print(f"not a zarr: {zarr}")
        return None
    else:
        layer = Layer(zarr)
        if not layer.specs:
            print(f"not an ome-zarr: {zarr}")
        LOGGER.debug(layer.data)
        return layer


def download(path: str, output_dir: str = ".", zarr_name: str = "") -> None:
    """
    download zarr from URL
    """
    layer = info(path)
    if not layer:
        return
    image_id = "unknown"
    resolutions: List[da.core.Array] = []
    datasets: List[str] = []
    for spec in layer.specs:
        if isinstance(spec, OMERO):
            image_id = spec.image_data.get("id", image_id)
        if isinstance(spec, Multiscales):
            datasets = spec.datasets
            resolutions = layer.data
            if not datasets or not resolutions:
                print("no multiscales data found")
                return

    LOGGER.info("image_id %s", image_id)
    if not zarr_name:
        zarr_name = f"{image_id}.zarr"

    target_dir = os.path.join(output_dir, f"{zarr_name}")
    if os.path.exists(target_dir):
        print(f"{target_dir} already exists!")
        return
    print(f"downloading to {target_dir}")

    pbar = ProgressBar()
    for dataset, data in reversed(list(zip(datasets, resolutions))):
        print("X", layer, dataset, data)
        LOGGER.info(f"resolution {dataset}...")
        with pbar:
            data.to_zarr(os.path.join(target_dir, dataset))

    with open(os.path.join(target_dir, ".zgroup"), "w") as f:
        f.write(json.dumps(layer.zarr.zgroup))
    with open(os.path.join(target_dir, ".zattrs"), "w") as f:
        metadata: JSONDict = {}
        layer.write_metadata(metadata)
        f.write(json.dumps(metadata))
