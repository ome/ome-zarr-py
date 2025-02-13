"""Utility methods for ome_zarr access."""

import csv
import json
import logging
import os
import urllib
import webbrowser
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from http.server import (  # type: ignore[attr-defined]
    HTTPServer,
    SimpleHTTPRequestHandler,
    test,
)
from pathlib import Path

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


def find_multiscales(path_to_zattrs):
    # return list of images. Each image is [path_to_zarr, name, dirname]
    # We want full path to find the multiscales Image. e.g. full/path/to/image.zarr/0
    # AND we want image Name, e.g. "image.zarr Series 0"
    # AND we want the dir path to use for Tags e.g. full/path/to
    with open(path_to_zattrs / ".zattrs") as f:
        text = f.read()
    zattrs = json.loads(text)
    if "plate" in zattrs:
        plate = zattrs.get("plate")
        wells = plate.get("wells")
        field = "0"
        print("well", wells[0])
        path_to_zarr = path_to_zattrs / wells[0].get("path") / field
        plate_name = os.path.basename(path_to_zattrs)
        return [[path_to_zarr, plate_name, os.path.dirname(path_to_zattrs)]]
    elif zattrs.get("bioformats2raw.layout") == 3:
        # Open OME/METADATA.ome.xml
        print("bioformats2raw.layout...")
        try:
            tree = ET.parse(path_to_zattrs / "OME" / "METADATA.ome.xml")
            root = tree.getroot()
            # spec says "If the "series" attribute does not exist and no "plate" is
            # present, separate "multiscales" images MUST be stored in consecutively
            # numbered groups starting from 0 (i.e. "0/", "1/", "2/", "3/", ...)."
            series = 0
            images = []
            for child in root:
                # tag is eg. {http://www.openmicroscopy.org/Schemas/OME/2016-06}Image
                if child.tag.endswith("Image"):
                    img_name = (
                        os.path.basename(path_to_zattrs) + " Series:" + str(series)
                    )
                    # Get Name from XML metadata, otherwise use path and Series
                    img_name = child.attrib.get("Name", img_name)
                    images.append(
                        [
                            path_to_zattrs / str(series),
                            img_name,
                            os.path.dirname(path_to_zattrs),
                        ]
                    )
                    series += 1
            return images
        except Exception as ex:
            print(ex)
    elif zattrs.get("multiscales"):
        return [
            [
                path_to_zattrs,
                os.path.basename(path_to_zattrs),
                os.path.dirname(path_to_zattrs),
            ]
        ]
    return []


def splitall(path):
    # Use os.path.split() repeatedly to split path into dirs
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def view(input_path: str, port: int = 8000) -> None:
    # serve the parent directory in a simple server with CORS. Open browser
    parent_dir, image_name = os.path.split(input_path)
    # in case input_path had trailing slash, we go one level up...
    if len(image_name) == 0:
        parent_dir, image_name = os.path.split(parent_dir)

    # walk the input path to find all .zattrs files...
    def walk(path: Path):
        print("walk", path, (path / ".zattrs").exists())
        if (path / ".zattrs").exists():
            yield from find_multiscales(path)
        else:
            for p in path.iterdir():
                if (p / ".zattrs").exists():
                    yield from find_multiscales(p)
                elif p.is_dir():
                    yield from walk(p)
                else:
                    continue

    zarrs = list(walk(Path(input_path)))

    for z in zarrs:
        # split file path into list
        z[2] = splitall(z[2])

    # If we have just one zarr, open ome-ngff-validator in a web browser...
    if len(zarrs) == 1:
        url = (
            f"https://ome.github.io/ome-ngff-validator/"
            f"?source=http://localhost:{port}/{image_name}"
        )
    elif len(zarrs) > 1:
        # ...otherwise write to CSV file and open in BioFile Finder
        max_folders = max(len(z[2]) for z in zarrs)
        col_names = ["File Path", "File Name"] + [
            f"Folder {i}" for i in range(max_folders)
        ]
        # open csv file and write lines...
        bff_csv = os.path.join(input_path, "biofile_finder.csv")
        with open(bff_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(col_names)
            for zarr_img in zarrs:
                # path_to_zarr, name, dirname
                file_path = f"http://localhost:{port}/{zarr_img[0]}"
                name = zarr_img[1]
                # folders list needs to be same length for every row.
                # e.g. ['f1', 'f2', '-', '-']
                folders = zarr_img[2] + ["-"] * (max_folders - len(zarr_img[2]))
                writer.writerow([file_path, name] + folders)

        source = {
            "uri": f"http://localhost:{port}/{image_name}/biofile_finder.csv",
            "type": "csv",
            "name": "biofile_finder.csv",
        }
        s = urllib.parse.quote(json.dumps(source))
        url = f"https://bff.allencell.org/app?source={s}"

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

    # Open in browser...
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
    nodes: list[Node] = list()
    paths: list[list[str]] = list()
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

        resolutions: list[da.core.Array] = []
        datasets: list[str] = []
        for spec in node.specs:
            if isinstance(spec, Multiscales):
                datasets = spec.datasets
                resolutions = node.data
                if datasets and resolutions:
                    pbar = ProgressBar()
                    for dataset, data in reversed(list(zip(datasets, resolutions))):
                        LOGGER.info("resolution %s...", dataset)
                        with pbar:
                            data.to_zarr(
                                str(target_path / dataset), dimension_separator="/"
                            )
            else:
                # Assume a group that needs metadata, like labels
                zarr.group(str(target_path))


def strip_common_prefix(parts: list[list[str]]) -> str:
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
