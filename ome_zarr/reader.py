"""Reading logic for ome-zarr."""

import logging
from abc import ABC
from string import ascii_uppercase
from typing import Any, Dict, Iterator, List, Optional, Type, Union, cast

import dask.array as da
import numpy as np
from dask import delayed
from vispy.color import Colormap

from .io import BaseZarrLocation
from .types import JSONDict

LOGGER = logging.getLogger("ome_zarr.reader")


class Node:
    """Container for a representation of the binary data somewhere in the data
    hierarchy."""

    def __init__(
        self,
        zarr: BaseZarrLocation,
        root: Union["Node", "Reader", List[BaseZarrLocation]],
        visibility: bool = True,
    ):
        self.zarr = zarr
        self.root = root
        self.seen: List[BaseZarrLocation] = []
        if isinstance(root, Node) or isinstance(root, Reader):
            self.seen = root.seen
        else:
            self.seen = cast(List[BaseZarrLocation], root)
        self.__visible = visibility

        # Likely to be updated by specs
        self.metadata: JSONDict = dict()
        self.data: List[da.core.Array] = list()
        self.specs: List[Spec] = []
        self.pre_nodes: List[Node] = []
        self.post_nodes: List[Node] = []

        # TODO: this should be some form of plugin infra over subclasses
        if Labels.matches(zarr):
            self.specs.append(Labels(self))
        if Label.matches(zarr):
            self.specs.append(Label(self))
        if Multiscales.matches(zarr):
            self.specs.append(Multiscales(self))
        if OMERO.matches(zarr):
            self.specs.append(OMERO(self))
        if Plate.matches(zarr):
            self.specs.append(Plate(self))

    @property
    def visible(self) -> bool:
        """True if this node should be displayed by default.

        An invisible node may have been requested by the instrument, by the
        user, or by the ome_zarr library after determining that this node
        is lower priority, e.g. to prevent too many nodes from being shown
        at once.
        """
        return self.__visible

    @visible.setter
    def visible(self, visibility: bool) -> bool:
        """
        Set the visibility for this node, returning the previous value.

        A change of the visibility will propagate to all subnodes.
        """
        old = self.__visible
        if old != visibility:
            self.__visible = visibility
            for node in self.pre_nodes + self.post_nodes:
                node.visible = visibility
        return old

    def load(self, spec_type: Type["Spec"]) -> Optional["Spec"]:
        for spec in self.specs:
            if isinstance(spec, spec_type):
                return spec
        return None

    def add(
        self,
        zarr: BaseZarrLocation,
        prepend: bool = False,
        visibility: Optional[bool] = None,
    ) -> "Optional[Node]":
        """Create a child node if this location has not yet been seen.

        Newly created nodes may be considered higher or lower priority than
        the current node, and may be set to invisible if necessary.

        :param zarr: Location in the node hierarchy to be added
        :param prepend: Whether the newly created node should be given higher
            priority than the current node, defaults to False
        :param visibility: Allows setting the node (and therefore layer)
            as deactivated for initial display or if None the value of the
            current node will be propagated to the new node,  defaults to None
        :return: Newly created node if this is the first time it has been
            encountered; None if the node has already been processed.
        """

        if zarr in self.seen:
            LOGGER.debug(f"already seen {zarr}; stopping recursion")
            return None

        if visibility is None:
            visibility = self.visible

        self.seen.append(zarr)
        node = Node(zarr, self, visibility=visibility)
        if prepend:
            self.pre_nodes.append(node)
        else:
            self.post_nodes.append(node)

        return node

    def write_metadata(self, metadata: JSONDict) -> None:
        for spec in self.specs:
            metadata.update(self.zarr.root_attrs)

    def __repr__(self) -> str:
        suffix = ""
        if not self.visible:
            suffix += " (hidden)"
        return f"{self.zarr}{suffix}"


class Spec(ABC):
    """Base class for specifications that can be implemented by groups or arrays within
    the hierarchy.

    Multiple subclasses may apply.
    """

    @staticmethod
    def matches(zarr: BaseZarrLocation) -> bool:
        raise NotImplementedError()

    def __init__(self, node: Node) -> None:
        self.node = node
        self.zarr = node.zarr
        LOGGER.debug(f"treating {self.zarr} as {self.__class__.__name__}")
        for k, v in self.zarr.root_attrs.items():
            LOGGER.info("root_attr: %s", k)
            LOGGER.debug(v)

    def lookup(self, key: str, default: Any) -> Any:
        return self.zarr.root_attrs.get(key, default)


class Labels(Spec):
    """Relatively small specification for the well-known "labels" group which only
    contains the name of subgroups which should be loaded as labeled images."""

    @staticmethod
    def matches(zarr: BaseZarrLocation) -> bool:
        """Does the Zarr Image group also include a /labels sub-group?"""
        # TODO: also check for "labels" entry and perhaps version?
        return bool("labels" in zarr.root_attrs)

    def __init__(self, node: Node) -> None:
        super().__init__(node)
        label_names = self.lookup("labels", [])
        for name in label_names:
            child_zarr = self.zarr.create(name)
            if child_zarr.exists():
                node.add(child_zarr)


class Label(Spec):
    """An additional aspect to a multiscale image is that it can be a labeled image, in
    which each discrete pixel value represents a separate object."""

    @staticmethod
    def matches(zarr: BaseZarrLocation) -> bool:
        """If label-specific metadata is present, then return true."""
        return bool("image-label" in zarr.root_attrs)

    def __init__(self, node: Node) -> None:
        super().__init__(node)

        image_label = self.lookup("image-label", {})

        image = image_label.get("source", {}).get("image", None)
        parent_zarr = None
        if image:
            # This is an ome mask, load the image
            parent_zarr = self.zarr.create(image)
            if parent_zarr.exists():
                LOGGER.debug(f"delegating to parent image: {parent_zarr}")
                node.add(parent_zarr, prepend=True, visibility=False)
            else:
                parent_zarr = None
        if parent_zarr is None:
            LOGGER.warn(f"no parent found for {self}: {image}")

        # Metadata: TODO move to a class
        colors: Dict[Union[int, bool], List[float]] = {}
        color_list = image_label.get("colors", [])
        if color_list:
            for color in color_list:
                try:
                    label_value = color["label-value"]
                    rgba = color.get("rgba", None)
                    if rgba:
                        rgba = [x / 255 for x in rgba]

                    if isinstance(label_value, bool) or isinstance(label_value, int):
                        colors[label_value] = rgba
                    else:
                        raise Exception("not bool or int")

                except Exception as e:
                    LOGGER.error(f"invalid color - {color}: {e}")

        # TODO: a metadata transform should be provided by specific impls.
        name = self.zarr.basename()
        node.metadata.update(
            {
                "visible": node.visible,
                "name": name,
                "color": colors,
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

    def __init__(self, node: Node) -> None:
        super().__init__(node)

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
            node.data.append(data)

        # Load possible node data
        child_zarr = self.zarr.create("labels")
        if child_zarr.exists():
            node.add(child_zarr, visibility=False)


class OMERO(Spec):
    @staticmethod
    def matches(zarr: BaseZarrLocation) -> bool:
        return bool("omero" in zarr.root_attrs)

    def __init__(self, node: Node) -> None:
        super().__init__(node)
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
            names: List[str] = [("channel_%d" % idx) for idx, ch in enumerate(channels)]
            visibles: List[bool] = [True for x in channels]

            for idx, ch in enumerate(channels):
                # 'FF0000' -> [1, 0, 0]

                color = ch.get("color", None)
                if color is not None:
                    rgb = [(int(color[i : i + 2], 16) / 255) for i in range(0, 6, 2)]
                    # TODO: make this value an enumeration
                    if model == "greyscale":
                        rgb = [1, 1, 1]
                    colormaps.append(Colormap([[0, 0, 0], rgb]))

                label = ch.get("label", None)
                if label is not None:
                    names[idx] = label

                visible = ch.get("active", None)
                if visible is not None:
                    visibles[idx] = visible and node.visible

                window = ch.get("window", None)
                if window is not None:
                    start = window.get("start", None)
                    end = window.get("end", None)
                    if start is None or end is None:
                        # Disable contrast limits settings if one is missing
                        contrast_limits = None
                    elif contrast_limits is not None:
                        contrast_limits[idx] = [start, end]

            node.metadata["name"] = names
            node.metadata["visible"] = visibles
            node.metadata["contrast_limits"] = contrast_limits
            node.metadata["colormap"] = colormaps
        except Exception as e:
            LOGGER.error(f"failed to parse metadata: {e}")


class Plate(Spec):
    @staticmethod
    def matches(zarr: BaseZarrLocation) -> bool:
        return bool("plate" in zarr.root_attrs)

    def __init__(self, node: Node) -> None:
        super().__init__(node)
        self.get_pyramid_lazy(node)

    def get_pyramid_lazy(self, node: Node) -> None:
        """
        Return a pyramid of dask data, where the highest resolution is the
        stitched full-resolution images.
        """
        self.plate_data = self.lookup("plate", {})
        print("self.plate_data", self.plate_data)
        self.rows = self.plate_data.get("rows", 0)
        self.cols = self.plate_data.get("columns", 0)
        self.acquisitions = self.plate_data.get('plateAcquisitions', [{"path": "0"}])

        self.fields = [
            "Field_1",
        ]
        self.row_labels = ascii_uppercase[0 : self.rows]
        self.col_labels = range(1, self.cols + 1)

        # TODO: support more Acquisitions - just 1st for now
        run = self.acquisitions[0]['path']
        rows = self.rows
        cols = self.cols
        row_labels = self.row_labels

        # Get the first image...
        path = f"{run}/{row_labels[0]}/{self.col_labels[0]}/{self.fields[0]}"
        image_zarr = self.zarr.create(path)
        image_node = Node(image_zarr, node)

        numpy_type = image_node.data[0].dtype
        img_shape = image_node.data[0].shape

        img_pyramid_shapes = [d.shape for d in image_node.data]

        print('img_pyramid_shapes', img_pyramid_shapes)
        size_t = img_shape[0]
        size_c = img_shape[1]
        size_z = img_shape[2]

        size_y = img_shape[3]
        size_x = img_shape[4]

        # FIXME - if only returning a single stiched plate (not a pyramid)
        # need to decide optimal size. E.g. longest side < 3000
        TARGET_SIZE = 1500
        plate_width = cols * size_x
        plate_height = cols * size_y
        longest_side = max(plate_width, plate_height)
        target_level = 0
        for level, shape in enumerate(img_pyramid_shapes):
            plate_width = cols * shape[-1]
            plate_height = rows * shape[-2]
            longest_side = max(plate_width, plate_height)
            target_level = level
            if longest_side <= TARGET_SIZE:
                break

        print("target_level", target_level)

        def get_tile(tile_name: str) -> np.ndarray:
            """ tile_name is 'level,z,c,t,row,col' """
            level, row, col = [int(n) for n in tile_name.split(",")]
            path = f"{run}/{row_labels[row]}/{col + 1}/Field_1/{level}"
            print(f"LOADING tile... {path}")

            data = self.zarr.load(path)
            print("...data.shape", data.shape)
            return data

        lazy_reader = delayed(get_tile)

        def get_lazy_plate(level: int) -> da.Array:
            lazy_rows = []
            # For level 0, return whole image for each tile
            for row in range(rows):
                lazy_row: List[da.Array] = []
                for col in range(cols):
                    tile_name = f"{level},{row},{col}"
                    print(f"creating lazy_reader. level:{level} row:{row} col:{col}")
                    tile_size = img_pyramid_shapes[level]
                    lazy_tile = da.from_delayed(
                        lazy_reader(tile_name), shape=tile_size, dtype=numpy_type
                    )
                    lazy_row.append(lazy_tile)
                lazy_rows.append(da.concatenate(lazy_row, axis=4))
                print("lazy_row.shape", lazy_rows[-1].shape)
            return da.concatenate(lazy_rows, axis=3)

        pyramid = []

        # This should create a pyramid of levels, but causes seg faults!
        # for level in range(5):
        for level in [target_level]:
            lazy_plate = get_lazy_plate(level)
            print("lazy_plate", lazy_plate.shape)
            pyramid.append(lazy_plate)

        # Set the node.data to be pyramid view of the plate
        node.data = pyramid
        # Use the first image's metadata for viewing the whole Plate
        node.metadata = image_node.metadata

        node.metadata.update({"metadata": {"plate": {"rows": rows, "columns": cols,}}})


class Reader:
    """Parses the given Zarr instance into a collection of Nodes properly ordered
    depending on context.

    Depending on the starting point, metadata may be followed up or down the Zarr
    hierarchy.
    """

    def __init__(self, zarr: BaseZarrLocation) -> None:
        assert zarr.exists()
        self.zarr = zarr
        self.seen: List[BaseZarrLocation] = [zarr]

    def __call__(self) -> Iterator[Node]:
        node = Node(self.zarr, self)
        if node.specs:  # Something has matched

            LOGGER.debug(f"treating {self.zarr} as ome-zarr")
            yield from self.descend(node)

            # TODO: API thoughts for the Spec type
            # - ask for recursion or not
            # - ask for "provides data", "overrides data"

        elif self.zarr.zarray:  # Nothing has matched
            LOGGER.debug(f"treating {self.zarr} as raw zarr")
            node.data.append(self.zarr.load())
            yield node

        else:
            LOGGER.debug(f"ignoring {self.zarr}")
            # yield nothing

    def descend(self, node: Node, depth: int = 0) -> Iterator[Node]:

        for pre_node in node.pre_nodes:
            yield from self.descend(pre_node, depth + 1)

        LOGGER.debug(f"returning {node}")
        yield node

        for post_node in node.post_nodes:
            yield from self.descend(post_node, depth + 1)
