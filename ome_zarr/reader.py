"""Reading logic for ome-zarr."""

import logging
from abc import ABC
from typing import Any, Dict, Iterator, List, Optional, Union, cast

import dask.array as da
from vispy.color import Colormap

from .conversions import int_to_rgba
from .io import BaseZarrLocation
from .types import JSONDict

LOGGER = logging.getLogger("ome_zarr.reader")


class Layer:
    """Container for a representation of the binary data somewhere in the data
    hierarchy."""

    def __init__(
        self, zarr: BaseZarrLocation, root: Union["Layer", "Reader", List[str]]
    ):
        self.zarr = zarr
        self.root = root
        self.seen: List[str] = []
        if isinstance(root, Layer) or isinstance(root, Reader):
            self.seen = root.seen
        else:
            self.seen = cast(List[str], root)
        self.visible = True

        # Likely to be updated by specs
        self.metadata: JSONDict = dict()
        self.data: List[da.core.Array] = list()
        self.specs: List[Spec] = []
        self.pre_layers: List[Layer] = []
        self.post_layers: List[Layer] = []

        # TODO: this should be some form of plugin infra over subclasses
        if Labels.matches(zarr):
            self.specs.append(Labels(self))
        if Label.matches(zarr):
            self.specs.append(Label(self))
        if Multiscales.matches(zarr):
            self.specs.append(Multiscales(self))
        if OMERO.matches(zarr):
            self.specs.append(OMERO(self))

    def add(self, zarr: BaseZarrLocation, prepend: bool = False,) -> "Optional[Layer]":
        """Create a child layer if this location has not yet been seen; otherwise return
        None."""

        if zarr.zarr_path in self.seen:
            LOGGER.debug(f"already seen {zarr}; stopping recursion")
            return None

        self.seen.append(zarr.zarr_path)
        layer = Layer(zarr, self)
        if prepend:
            self.pre_layers.append(layer)
        else:
            self.post_layers.append(layer)

        return layer

    def write_metadata(self, metadata: JSONDict) -> None:
        for spec in self.specs:
            metadata.update(self.zarr.root_attrs)

    def __repr__(self) -> str:
        suffix = ""
        if self.zarr.zgroup:
            suffix += " [zgroup]"
        if self.zarr.zarray:
            suffix += " [zarray]"
        return f"{self.zarr.zarr_path}{suffix}"


class Spec(ABC):
    """Base class for specifications that can be implemented by groups or arrays within
    the hierarchy.

    Multiple subclasses may apply.
    """

    @staticmethod
    def matches(zarr: BaseZarrLocation) -> bool:
        raise NotImplementedError()

    def __init__(self, layer: Layer) -> None:
        self.layer = layer
        self.zarr = layer.zarr
        LOGGER.debug(f"treating {self.zarr} as {self.__class__.__name__}")
        for k, v in self.zarr.root_attrs.items():
            LOGGER.info("root_attr: %s", k)
            LOGGER.debug(v)

    def lookup(self, key: str, default: Any) -> Any:
        return self.zarr.root_attrs.get(key, default)


class Labels(Spec):
    """Relatively small specification for the well-known "labels" group which only
    contains the name of subgroups which should be loaded an labeled images."""

    @staticmethod
    def matches(zarr: BaseZarrLocation) -> bool:
        """Does the Zarr Image group also include a /labels sub-group?"""
        # TODO: also check for "labels" entry and perhaps version?
        return bool("labels" in zarr.root_attrs)

    def __init__(self, layer: Layer) -> None:
        super().__init__(layer)
        label_names = self.lookup("labels", [])
        for name in label_names:
            child_zarr = self.zarr.create(name)
            if child_zarr.exists():
                layer.add(child_zarr)


class Label(Spec):
    """An additional aspect to a multiscale image is that it can be a labeled image, in
    which each discrete pixel value represents a separate object."""

    @staticmethod
    def matches(zarr: BaseZarrLocation) -> bool:
        """If label-specific metadata is present, then return true."""
        # FIXME: this should be the "label" metadata soon
        return bool("colors" in zarr.root_attrs or "image" in zarr.root_attrs)

    def __init__(self, layer: Layer) -> None:
        super().__init__(layer)
        layer.visible = True

        image = self.lookup("image", {}).get("array", None)
        parent_zarr = None
        if image:
            # This is an ome mask, load the image
            parent_zarr = self.zarr.create(image)
            if parent_zarr.exists():
                LOGGER.debug(f"delegating to parent image: {parent_zarr}")
                parent_layer = layer.add(parent_zarr, prepend=True)
                if parent_layer is not None:
                    layer.visible = False
            else:
                parent_zarr = None
        if parent_zarr is None:
            LOGGER.warn(f"no parent found for {self}: {image}")

        # Metadata: TODO move to a class
        colors: Dict[Union[int, bool], List[float]] = {}
        color_dict = self.lookup("color", {})
        if color_dict:
            for k, v in color_dict.items():
                try:
                    if k.lower() == "true":
                        k = True
                    elif k.lower() == "false":
                        k = False
                    else:
                        k = int(k)
                    colors[k] = int_to_rgba(v)
                except Exception as e:
                    LOGGER.error(f"invalid color - {k}={v}: {e}")

        # TODO: a metadata transform should be provided by specific impls.
        name = self.zarr.zarr_path.split("/")[-1]
        layer.metadata.update(
            {
                "visible": False,
                "name": name,
                "colormap": colors,
                "metadata": {"image": self.lookup("image", {}), "path": name},
            }
        )


class Multiscales(Spec):
    @staticmethod
    def matches(zarr: BaseZarrLocation) -> bool:
        """is multiscales metadata present?"""
        if zarr.zgroup:
            if "multiscales" in zarr.root_attrs:
                return True
        return False

    def __init__(self, layer: Layer) -> None:
        super().__init__(layer)

        try:
            datasets = self.lookup("multiscales", [])[0]["datasets"]
            datasets = [d["path"] for d in datasets]
            self.datasets: List[str] = datasets
            LOGGER.info("datasets %s", datasets)
        except Exception as e:
            LOGGER.error(f"failed to parse multiscale metadata: {e}")
            return  # EARLY EXIT

        for resolution in self.datasets:
            # data.shape is (t, c, z, y, x) by convention
            data: da.core.Array = self.zarr.load(resolution)
            chunk_sizes = [
                str(c[0]) + (" (+ %s)" % c[-1] if c[-1] != c[0] else "")
                for c in data.chunks
            ]
            LOGGER.info("resolution: %s", resolution)
            LOGGER.info(" - shape (t, c, z, y, x) = %s", data.shape)
            LOGGER.info(" - chunks =  %s", chunk_sizes)
            LOGGER.info(" - dtype = %s", data.dtype)
            layer.data.append(data)

        # TODO: test removal
        if len(layer.data) == 1:
            layer.data = layer.data[0]

        # Load possible layer data
        child_zarr = self.zarr.create("labels")
        if child_zarr.exists():
            layer.add(child_zarr)


class OMERO(Spec):
    @staticmethod
    def matches(zarr: BaseZarrLocation) -> bool:
        return bool("omero" in zarr.root_attrs)

    def __init__(self, layer: Layer) -> None:
        super().__init__(layer)
        # TODO: start checking metadata version
        self.image_data = self.lookup("omero", {})

        try:
            model = "unknown"
            rdefs = self.image_data.get("rdefs", {})
            if rdefs:
                model = rdefs.get("model", "unset")

            channels = self.image_data.get("channels", None)
            if channels is None:
                return  # EARLY EXIT

            try:
                len(channels)
            except Exception:
                LOGGER.warn(f"error counting channels: {channels}")
                return  # EARLY EXIT

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

            layer.metadata["colormap"] = colormaps
            layer.metadata["contrast_limits"] = contrast_limits
            layer.metadata["name"] = names
            layer.metadata["visible"] = visibles
        except Exception as e:
            LOGGER.error(f"failed to parse metadata: {e}")


class Reader:
    """Parses the given Zarr instance into a collection of Layers properly ordered
    depending on context.

    Depending on the starting point, metadata may be followed up or down the Zarr
    hierarchy.
    """

    def __init__(self, zarr: BaseZarrLocation) -> None:
        assert zarr.is_zarr()
        self.zarr = zarr
        self.seen: List[str] = [zarr.zarr_path]

    def __call__(self) -> Iterator[Layer]:
        layer = Layer(self.zarr, self)
        if layer.specs:  # Something has matched

            LOGGER.debug(f"treating {self.zarr} as ome-zarr")
            yield from self.descend(layer)

            # TODO: API thoughts for the Spec type
            # - ask for earlier_layers, later_layers (i.e. priorities)
            # - ask for recursion or not
            # - ask for visible or invisible (?)
            # - ask for "provides data", "overrides data"

        elif self.zarr.zarray:  # Nothing has matched
            LOGGER.debug(f"treating {self.zarr} as raw zarr")
            data = da.from_zarr(f"{self.zarr.zarr_path}")
            layer.data.append(data)
            yield layer

        else:
            LOGGER.debug(f"ignoring {self.zarr}")
            # yield nothing

    def descend(self, layer: Layer, depth: int = 0) -> Iterator[Layer]:

        for pre_layer in layer.pre_layers:
            yield from self.descend(pre_layer, depth + 1)

        LOGGER.debug(f"returning {layer}")
        yield layer

        for post_layer in layer.post_layers:
            yield from self.descend(post_layer, depth + 1)
