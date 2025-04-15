"""Utility methods for ome_zarr access."""

import csv
import json
import logging
import os
import urllib
import webbrowser
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import datetime
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

    Parameters
    ----------
    path :
        Path to OME-Zarr fileset.
    stats :
        If True, print stats (currently just minimum/maximum of all arrays)

    Warnings
    --------
    Passing ``stats=True`` will trigger a full read of every array in the fileset.
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


def view(input_path: str, port: int = 8000, dry_run: bool = False) -> None:
    # serve the parent directory in a simple server with CORS. Open browser
    # dry_run is for testing, so we don't open the browser or start the server

    zarrs = []
    if (Path(input_path) / ".zattrs").exists():
        zarrs = find_multiscales(Path(input_path))
    if len(zarrs) == 0:
        print(
            f"No OME-Zarr images found in {input_path}. "
            f"Try $ ome_zarr finder {input_path}"
        )
        return

    parent_dir, image_name = os.path.split(input_path)
    if len(image_name) == 0:
        parent_dir, image_name = os.path.split(parent_dir)
    parent_dir = str(parent_dir)

    # open ome-ngff-validator in a web browser...
    url = (
        f"https://ome.github.io/ome-ngff-validator/"
        f"?source=http://localhost:{port}/{image_name}"
    )

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

    # for testing
    if dry_run:
        return

    # Open in browser...
    webbrowser.open(url)

    # ...then start serving content
    test(CORSRequestHandler, HTTPServer, port=port)


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
        if len(wells) > 0:
            path_to_zarr = path_to_zattrs / wells[0].get("path") / field
            plate_name = os.path.basename(path_to_zattrs)
            return [[path_to_zarr, plate_name, os.path.dirname(path_to_zattrs)]]
        else:
            LOGGER.info(f"No wells found in plate{path_to_zattrs}")
            return []
    elif zattrs.get("bioformats2raw.layout") == 3:
        # Open OME/METADATA.ome.xml
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


def finder(input_path: str, port: int = 8000, dry_run=False) -> None:
    # serve the parent directory in a simple server with CORS. Open browser
    # dry_run is for testing, so we don't open the browser or start the server
    parent_path, server_dir = os.path.split(input_path)
    # in case input_path had trailing slash, we go one level up...
    if len(server_dir) == 0:
        parent_path, server_dir = os.path.split(parent_path)

    # 'input_path' is path passed to the script. To the data dir. E.g. "ZARR/data"
    # 'parent_path', e.g. "ZARR" just for running http server
    # 'server_dir' is the name of our top-level dir E.g. "data"

    # We will be serving the data from last dir in /parent/dir/path
    # so we need to use that as base for image URLs...

    # walk the input path to find all .zattrs files...
    def walk(path: Path):
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

    url = None
    zarrs = list(walk(Path(input_path)))

    # If we have just one zarr, open ome-ngff-validator in a web browser...
    if len(zarrs) == 0:
        print("No OME-Zarr files found in", input_path)
        return
    else:
        # ...otherwise write to CSV file and open in BioFile Finder
        col_names = ["File Path", "File Name", "Folders", "Uploaded"]
        # write csv file into the dir we're serving from...
        bff_csv = os.path.join(input_path, "biofile_finder.csv")

        with open(bff_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(col_names)
            for zarr_img in zarrs:
                # zarr paths start with full path to img
                # e.g. ZARR/data/to/img (from walk("ZARR/data"))
                # but we want them to be from the server_dir to img, e.g "data/to/img".
                # So we want relative /to/img path, from input_path -> to img
                relpath = os.path.relpath(zarr_img[0], input_path)
                # On Windows, we need to replace \\ with / in relpath for URL
                rel_url = "/".join(splitall(relpath))
                file_path = f"http://localhost:{port}/{server_dir}/{rel_url}"
                name = zarr_img[1] or os.path.basename(zarr_img[0])
                # folders is "f1,f2,f3" etc.
                folders_path = os.path.relpath(zarr_img[2], input_path)
                folders = ",".join(splitall(folders_path))
                timestamp = ""
                try:
                    mtime = os.path.getmtime(zarr_img[0])
                    # format mtime as "YYYY-MM-DD HH:MM:SS.Z"
                    timestamp = datetime.fromtimestamp(mtime).strftime(
                        "%Y-%m-%d %H:%M:%S.%Z"
                    )
                except OSError:
                    pass
                writer.writerow([file_path, name, folders, timestamp])

        source = {
            "uri": f"http://localhost:{port}/{server_dir}/biofile_finder.csv",
            "type": "csv",
            "name": "biofile_finder.csv",
        }
        s = urllib.parse.quote(json.dumps(source))
        url = f"https://bff.allencell.org/app?source={s}"
        # show small thumbnails view by default. (v=3 for big thumbnails)
        url += "&v=2"

    class CORSRequestHandler(SimpleHTTPRequestHandler):
        def end_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            SimpleHTTPRequestHandler.end_headers(self)

        def translate_path(self, path: str) -> str:
            # Since we don't call the class constructor ourselves,
            # we set the directory here instead
            self.directory = parent_path
            super_path = super().translate_path(path)
            return super_path

    # for testing
    if dry_run:
        return

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
