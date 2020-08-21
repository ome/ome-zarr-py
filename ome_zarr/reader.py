"""
Reading logic for ome-zarr
"""
import json
import logging
import os
import posixpath
import warnings

# for optional type hints only, otherwise you can delete/ignore this stuff
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import dask.array as da
import requests
from vispy.color import Colormap

LOGGER = logging.getLogger("ome_zarr.reader")

# START type hint stuff
LayerData = Union[Tuple[Any], Tuple[Any, Dict], Tuple[Any, Dict, str]]
PathLike = Union[str, List[str]]
ReaderFunction = Callable[[PathLike], List[LayerData]]
# END type hint stuff.


class BaseZarr:
    def __init__(self, path: str) -> None:
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

    def __str__(self) -> str:
        suffix = ""
        if self.zgroup:
            suffix += " [zgroup]"
        if self.zarray:
            suffix += " [zarray]"
        return f"{self.zarr_path}{suffix}"

    def is_zarr(self) -> Optional[Dict]:
        return self.zarray or self.zgroup

    def is_ome_zarr(self) -> bool:
        return bool(self.zgroup) and "multiscales" in self.root_attrs

    def has_ome_labels(self) -> Dict:
        "Does the zarr Image also include /labels sub-dir"
        return self.get_json("labels/.zgroup")

    def is_ome_labels_group(self) -> bool:
        # TODO: also check for "labels" entry and perhaps version?
        return self.zarr_path.endswith("labels/") and bool(self.get_json(".zgroup"))

    def get_label_names(self) -> List[str]:
        """
        Called if is_ome_label is true
        """
        # If this is a label, the names are in root .zattrs
        return self.root_attrs.get("labels", [])

    def get_json(self, subpath: str) -> Dict:
        raise NotImplementedError("unknown")

    def get_reader_function(self) -> Callable:
        if not self.is_zarr():
            raise Exception(f"not a zarr: {self}")
        return self.reader_function

    def to_rgba(self, v: int) -> List[float]:
        """Get rgba (0-1) e.g. (1, 0.5, 0, 1) from integer"""
        return [x / 255 for x in v.to_bytes(4, signed=True, byteorder="big")]

    def update_metadata(self, data: LayerData, **kwargs: Any) -> LayerData:
        """Cast LayerData for setting metadata"""
        # Union[Tuple[Any], Tuple[Any, Dict], Tuple[Any, Dict, str]]
        if not data:
            return None
        elif len(data) == 1:  # Tuple[Any]
            return (data[0], dict(kwargs))
        else:
            data = cast(Tuple[Any, Dict], data)
            data[1].update(kwargs)
        return data

    def new_reader(self, path: str, recurse: bool = False) -> Optional[List[LayerData]]:
        """Create a new reader for the given path"""
        return self.__class__(path).reader_function(None, recurse=recurse)

    def reader_function(
        self, path: Optional[PathLike], recurse: bool = True,
    ) -> Optional[List[LayerData]]:
        """Take a path or list of paths and return a list of LayerData tuples."""

        if isinstance(path, list):
            path = path[0]
            # TODO: safe to ignore this path?

        if self.is_ome_zarr():
            LOGGER.debug(f"treating {path} as ome-zarr")
            layers = [self.load_ome_zarr()]
            # If the Image contains labels...
            if recurse and self.has_ome_labels():
                labels_path = os.path.join(self.zarr_path, "labels")
                # Create a new OME Zarr Reader to load labels
                labels = self.new_reader(labels_path)
                if labels:
                    layers.extend(labels)
            return layers

        elif self.is_ome_labels_group():

            LOGGER.debug(f"treating {path} as labels")
            label_names = self.get_label_names()
            rv: List[LayerData] = []
            for name in label_names:

                # Load multiscale as well as label metadata
                label_path: str = os.path.join(self.zarr_path, name)
                multiscales: Optional[List[LayerData]] = self.new_reader(label_path)
                if not multiscales:
                    continue
                metadata: Dict = self.load_ome_label_metadata(name)

                # Look parent
                path = metadata.get("path", None)
                image = metadata.get("image", {}).get("array", None)
                if recurse and path and image:
                    # This is an ome mask, load the image
                    parent = posixpath.normpath(f"{path}/{image}")
                    LOGGER.debug(f"delegating to parent image: {parent}")
                    # Create a new OME Zarr Reader to load labels
                    replace = self.new_reader(parent)
                    if replace:
                        # Set replacements to be invisible
                        for r in replace:
                            r = self.update_metadata(r, visible=False)
                        rv.extend(replace)
                for multiscale in multiscales:
                    multiscale = self.update_metadata(multiscale, visible=True)
                rv.extend(multiscales)
            return rv

        # TODO: might also be an individiaul mask

        elif self.zarray:
            LOGGER.debug(f"treating {path} as raw zarr")
            data = da.from_zarr(f"{self.zarr_path}")
            return [(data,)]

        else:
            LOGGER.debug(f"ignoring {path}")
            return None

    def load_omero_metadata(self, assert_channel_count: int = None) -> Dict:
        """Load OMERO metadata as json and convert for napari"""
        metadata: Dict[str, Any] = {}
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
            contrast_limits: Optional[List[Optional[Any]]] = [None for x in channels]
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

    def load_ome_zarr(self) -> LayerData:

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

    def load_ome_label_metadata(self, name: str) -> Dict:
        # Metadata: TODO move to a class
        label_attrs = self.get_json(f"{name}/.zattrs")
        colors: Dict[Union[int, bool], List[float]] = {}
        color_dict = label_attrs.get("color", {})
        if color_dict:
            for k, v in color_dict.items():
                try:
                    if k.lower() == "true":
                        k = True
                    elif k.lower() == "false":
                        k = False
                    else:
                        k = int(k)
                    colors[k] = self.to_rgba(v)
                except Exception as e:
                    LOGGER.error(f"invalid color - {k}={v}: {e}")
        return {
            "visible": False,
            "name": name,
            "color": colors,
            "metadata": {"image": label_attrs.get("image", {}), "path": name},
        }


class LocalZarr(BaseZarr):
    def get_json(self, subpath: str) -> Dict:
        filename = os.path.join(self.zarr_path, subpath)

        if not os.path.exists(filename):
            return {}

        with open(filename) as f:
            return json.loads(f.read())


class RemoteZarr(BaseZarr):
    def get_json(self, subpath: str) -> Dict:
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
