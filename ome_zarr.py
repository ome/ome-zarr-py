"""
This module is a napari plugin.

It implements the ``napari_get_reader`` hook specification, (to create
a reader plugin).

Type annotations here are OPTIONAL!
If you don't care to annotate the return types of your functions
your plugin doesn't need to import, or even depend on napari at all!

Replace code below accordingly.
"""
import os
import json
import requests
import dask.array as da
import warnings

from dask.diagnostics import ProgressBar
from vispy.color import Colormap

from urllib.parse import urlparse


try:
    from napari_plugin_engine import napari_hook_implementation
except ImportError:

    def napari_hook_implementation(func, *args, **kwargs):
        return func


import logging

# for optional type hints only, otherwise you can delete/ignore this stuff
from typing import List, Optional, Union, Any, Tuple, Dict, Callable

LOGGER = logging.getLogger("ome_zarr")


LayerData = Union[Tuple[Any], Tuple[Any, Dict], Tuple[Any, Dict, str]]
PathLike = Union[str, List[str]]
ReaderFunction = Callable[[PathLike], List[LayerData]]
# END type hint stuff.


@napari_hook_implementation
def napari_get_reader(path: PathLike) -> Optional[ReaderFunction]:
    """
    Returns a reader for supported paths that include IDR ID

    - URL of the form: https://s3.embassy.ebi.ac.uk/idr/zarr/v0.1/ID.zarr/
    """
    if isinstance(path, list):
        path = path[0]
    instance = parse_url(path)
    if instance is not None and instance.is_zarr():
        return instance.get_reader_function()
    # Ignoring this path
    return None


def parse_url(path):
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


class BaseZarr:
    def __init__(self, path):
        self.zarr_path = path.endswith("/") and path or f"{path}/"
        self.zarray = self.get_json(".zarray")
        self.zgroup = self.get_json(".zgroup")
        if self.zgroup:
            self.root_attrs = self.get_json(".zattrs")
            if "omero" in self.root_attrs:
                self.image_data = self.root_attrs["omero"]
                # TODO: start checking metadata version
            else:
                # Backup location that can be removed in the future.
                warnings.warn("deprecated loading of omero.josn", DeprecationWarning)
                self.image_data = self.get_json("omero.json")

    def __str__(self):
        suffix = ""
        if self.zgroup:
            suffix += " [zgroup]"
        if self.zarray:
            suffix += " [zarray]"
        return f"{self.zarr_path}{suffix}"

    def is_zarr(self):
        return self.zarray or self.zgroup

    def is_ome_zarr(self):
        return self.zgroup and "multiscales" in self.root_attrs

    def has_ome_masks(self):
        "Does the zarr Image also include /masks sub-dir"
        return self.get_json("masks/.zgroup")

    def is_ome_mask(self):
        return self.zarr_path.endswith("masks/") and self.get_json(".zgroup")

    def get_mask_names(self):
        """
        Called if is_ome_mask is true
        """
        # If this is a mask, the names are in root .zattrs
        return self.root_attrs.get("masks", [])

    def get_json(self, subpath):
        raise NotImplementedError("unknown")

    def get_reader_function(self):
        if not self.is_zarr():
            raise Exception(f"not a zarr: {self}")
        return self.reader_function

    def to_rgba(self, v):
        """Get rgba (0-1) e.g. (1, 0.5, 0, 1) from integer"""
        return [x / 255 for x in v.to_bytes(4, signed=True, byteorder="big")]

    def reader_function(self, path: Optional[PathLike]) -> Optional[List[LayerData]]:
        """Take a path or list of paths and return a list of LayerData tuples."""

        if isinstance(path, list):
            path = path[0]
            # TODO: safe to ignore this path?

        if self.is_ome_zarr():
            LOGGER.debug(f"treating {path} as ome-zarr")
            layers = [self.load_ome_zarr()]
            # If the Image contains masks...
            if self.has_ome_masks():
                mask_path = os.path.join(self.zarr_path, "masks")
                # Create a new OME Zarr Reader to load masks
                masks = self.__class__(mask_path).reader_function(None)
                if masks:
                    layers.extend(masks)
            return layers

        elif self.zarray:
            LOGGER.debug(f"treating {path} as raw zarr")
            data = da.from_zarr(f"{self.zarr_path}")
            return [(data,)]

        elif self.is_ome_mask():
            LOGGER.debug(f"treating {path} as masks")
            return self.load_ome_masks()

        else:
            LOGGER.debug(f"ignoring {path}")
            return None

    def load_omero_metadata(self, assert_channel_count=None):
        """Load OMERO metadata as json and convert for napari"""
        metadata = {}
        try:
            model = "unknown"
            rdefs = self.image_data.get("rdefs", {})
            if rdefs:
                model = rdefs.get("model", "unset")

            channels = self.image_data.get("channels", None)
            if channels is None:
                return {}

            count = None
            try:
                count = len(channels)
                if assert_channel_count:
                    if count != assert_channel_count:
                        LOGGER.error(
                            (
                                "unexpected channel count: "
                                f"{count}!={assert_channel_count}"
                            )
                        )
                        return {}
            except Exception:
                LOGGER.warn(f"error counting channels: {channels}")
                return {}

            colormaps = []
            contrast_limits = [None for x in channels]
            names = [("channel_%d" % idx) for idx, ch in enumerate(channels)]
            visibles = [True for x in channels]

            for idx, ch in enumerate(channels):
                # 'FF0000' -> [1, 0, 0]

                color = ch.get("color", None)
                if color is not None:
                    rgb = [(int(color[i : i + 2], 16) / 255) for i in range(0, 6, 2)]
                    if model == "greyscale":
                        rgb = [1, 1, 1]
                    colormaps.append(Colormap([[0, 0, 0], rgb]))

                label = ch.get("label", None)
                if label is not None:
                    names[idx] = label

                visible = ch.get("active", None)
                if visible is not None:
                    visibles[idx] = visible

                window = ch.get("window", None)
                if window is not None:
                    start = window.get("start", None)
                    end = window.get("end", None)
                    if start is None or end is None:
                        # Disable contrast limits settings if one is missing
                        contrast_limits = None
                    elif contrast_limits is not None:
                        contrast_limits[idx] = [start, end]

            metadata["colormap"] = colormaps
            metadata["contrast_limits"] = contrast_limits
            metadata["name"] = names
            metadata["visible"] = visibles
        except Exception as e:
            LOGGER.error(f"failed to parse metadata: {e}")

        return metadata

    def load_ome_zarr(self):

        resolutions = ["0"]  # TODO: could be first alphanumeric dataset on err
        try:
            for k, v in self.root_attrs.items():
                LOGGER.info("root_attr: %s", k)
                LOGGER.debug(v)
            if "multiscales" in self.root_attrs:
                datasets = self.root_attrs["multiscales"][0]["datasets"]
                resolutions = [d["path"] for d in datasets]
        except Exception as e:
            raise e

        pyramid = []
        for resolution in resolutions:
            # data.shape is (t, c, z, y, x) by convention
            data = da.from_zarr(f"{self.zarr_path}{resolution}")
            chunk_sizes = [
                str(c[0]) + (" (+ %s)" % c[-1] if c[-1] != c[0] else "")
                for c in data.chunks
            ]
            LOGGER.info("resolution: %s", resolution)
            LOGGER.info(" - shape (t, c, z, y, x) = %s", data.shape)
            LOGGER.info(" - chunks =  %s", chunk_sizes)
            LOGGER.info(" - dtype = %s", data.dtype)
            pyramid.append(data)

        if len(pyramid) == 1:
            pyramid = pyramid[0]

        metadata = self.load_omero_metadata(data.shape[1])
        return (pyramid, {"channel_axis": 1, **metadata})

    def load_ome_masks(self):
        # look for masks in this dir...
        mask_names = self.get_mask_names()
        masks = []
        for name in mask_names:
            mask_path = os.path.join(self.zarr_path, name)
            mask_attrs = self.get_json(f"{name}/.zattrs")
            colors = {}
            if "color" in mask_attrs:
                color_dict = mask_attrs.get("color")
                colors = {int(k): self.to_rgba(v) for (k, v) in color_dict.items()}
            data = da.from_zarr(mask_path)
            # Split masks into separate channels, 1 per layer
            for n in range(data.shape[1]):
                masks.append(
                    (data[:, n, :, :, :], {"name": name, "color": colors}, "labels")
                )
        return masks


class LocalZarr(BaseZarr):
    def get_json(self, subpath):
        filename = os.path.join(self.zarr_path, subpath)

        if not os.path.exists(filename):
            return {}

        with open(filename) as f:
            return json.loads(f.read())


class RemoteZarr(BaseZarr):
    def get_json(self, subpath):
        url = f"{self.zarr_path}{subpath}"
        try:
            rsp = requests.get(url)
        except Exception:
            LOGGER.warn(f"unreachable: {url} -- details logged at debug")
            LOGGER.debug("exception details:", exc_info=True)
            return {}
        try:
            if rsp.status_code in (403, 404):  # file doesn't exist
                return {}
            return rsp.json()
        except Exception:
            LOGGER.error(f"({rsp.status_code}): {rsp.text}")
            return {}


def info(path):
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


def download(path, output_dir=".", zarr_name=""):
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
        datasets = [x["path"] for x in omezarr.root_attrs["multiscales"][0]["datasets"]]
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
