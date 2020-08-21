"""
Utility methods for ome_zarr access
"""

import json
import logging
import os
from urllib.parse import urlparse

import dask.array as da
from dask.diagnostics import ProgressBar

from .reader import BaseZarr, LocalZarr, RemoteZarr

LOGGER = logging.getLogger("ome_zarr.utils")


def parse_url(path: str) -> BaseZarr:
    # Check is path is local directory first
    if os.path.isdir(path):
        return LocalZarr(path)
    else:
        result = urlparse(path)
        if result.scheme in ("", "file"):
            # Strips 'file://' if necessary
            return LocalZarr(result.path)
        else:
            return RemoteZarr(path)


def info(path: str) -> None:
    """
    print information about the ome-zarr fileset
    """
    zarr = parse_url(path)
    if not zarr.is_ome_zarr():
        print(f"not an ome-zarr: {zarr}")
        return
    reader = zarr.get_reader_function()
    data = reader(path)
    LOGGER.debug(data)


def download(path: str, output_dir: str = ".", zarr_name: str = "") -> None:
    """
    download zarr from URL
    """
    omezarr = parse_url(path)
    if not omezarr.is_ome_zarr():
        print(f"not an ome-zarr: {path}")
        return

    image_id = omezarr.image_data.get("id", "unknown")
    LOGGER.info("image_id %s", image_id)
    if not zarr_name:
        zarr_name = f"{image_id}.zarr"

    try:
        datasets = omezarr.root_attrs["multiscales"][0]["datasets"]
        datasets = [x["path"] for x in datasets]
    except KeyError:
        datasets = ["0"]
    LOGGER.info("datasets %s", datasets)
    resolutions = [da.from_zarr(path, component=str(i)) for i in datasets]
    # levels = list(range(len(resolutions)))

    target_dir = os.path.join(output_dir, f"{zarr_name}")
    if os.path.exists(target_dir):
        print(f"{target_dir} already exists!")
        return
    print(f"downloading to {target_dir}")

    pbar = ProgressBar()
    for dataset, data in reversed(list(zip(datasets, resolutions))):
        LOGGER.info(f"resolution {dataset}...")
        with pbar:
            data.to_zarr(os.path.join(target_dir, dataset))

    with open(os.path.join(target_dir, ".zgroup"), "w") as f:
        f.write(json.dumps(omezarr.zgroup))
    with open(os.path.join(target_dir, ".zattrs"), "w") as f:
        f.write(json.dumps(omezarr.root_attrs))
