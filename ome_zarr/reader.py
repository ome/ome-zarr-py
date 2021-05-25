"""Reading logic for ome-zarr."""

import logging
import math
from abc import ABC
from typing import Any, Dict, Iterator, List, Optional, Type, Union, cast

import dask.array as da
import numpy as np
from dask import delayed
from vispy.color import Colormap

from .io import ZarrLocation
from .types import JSONDict

LOGGER = logging.getLogger("ome_zarr.reader")


class Node:
    """Container for a representation of the binary data somewhere in the data
    hierarchy."""

    def __init__(
        self,
        zarr: ZarrLocation,
        root: Union["Node", "Reader", List[ZarrLocation]],
        visibility: bool = True,
        plate_labels: bool = False,
    ):
        self.zarr = zarr
        self.root = root
        self.seen: List[ZarrLocation] = []
        if isinstance(root, Node) or isinstance(root, Reader):
            self.seen = root.seen
        else:
            self.seen = cast(List[ZarrLocation], root)
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
        if plate_labels:
            self.specs.append(PlateLabels(self))
        elif Plate.matches(zarr):
            self.specs.append(Plate(self))
            self.add(zarr, plate_labels=True)
        if Well.matches(zarr):
            self.specs.append(Well(self))

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
        zarr: ZarrLocation,
        prepend: bool = False,
        visibility: Optional[bool] = None,
        plate_labels: bool = False,
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

        if zarr in self.seen and not plate_labels:
            LOGGER.debug(f"already seen {zarr}; stopping recursion")
            return None

        if visibility is None:
            visibility = self.visible

        self.seen.append(zarr)
        node = Node(zarr, self, visibility=visibility, plate_labels=plate_labels)
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
    def matches(zarr: ZarrLocation) -> bool:
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
    def matches(zarr: ZarrLocation) -> bool:
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
    def matches(zarr: ZarrLocation) -> bool:
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

        properties: Dict[int, Dict[str, str]] = {}
        props_list = image_label.get("properties", [])
        if props_list:
            for props in props_list:
                label_val = props["label-value"]
                properties[label_val] = dict(props)
                del properties[label_val]["label-value"]

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
        if properties:
            node.metadata.update({"properties": properties})


class Multiscales(Spec):
    @staticmethod
    def matches(zarr: ZarrLocation) -> bool:
        """is multiscales metadata present?"""
        if zarr.zgroup:
            if "multiscales" in zarr.root_attrs:
                return True
        return False

    def __init__(self, node: Node) -> None:
        super().__init__(node)

        try:
            multiscales = self.lookup("multiscales", [])
            version = multiscales[0].get("version", "0.1")
            datasets = multiscales[0]["datasets"]
            datasets = [d["path"] for d in datasets]
            self.datasets: List[str] = datasets
            LOGGER.info("datasets %s", datasets)
        except Exception as e:
            LOGGER.error(f"failed to parse multiscale metadata: {e}")
            return  # EARLY EXIT

        for resolution in self.datasets:
            # data.shape is (t, c, z, y, x) by convention
            data: da.core.Array = self.array(resolution, version)
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

    def array(self, resolution: str, version: str) -> da.core.Array:
        nested = version != "0.1"
        # data.shape is (t, c, z, y, x) by convention
        return self.zarr.load(resolution, nested)


class OMERO(Spec):
    @staticmethod
    def matches(zarr: ZarrLocation) -> bool:
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


class Well(Spec):
    @staticmethod
    def matches(zarr: ZarrLocation) -> bool:
        return bool("well" in zarr.root_attrs)

    def __init__(self, node: Node) -> None:
        super().__init__(node)
        self.well_data = self.lookup("well", {})
        LOGGER.info("well_data: %s", self.well_data)

        image_paths = [image["path"] for image in self.well_data.get("images")]
        field_count = len(image_paths)
        column_count = math.ceil(math.sqrt(field_count))
        row_count = math.ceil(field_count / column_count)

        # Use first Field for rendering settings, shape etc.
        image_zarr = self.zarr.create(image_paths[0])
        image_node = Node(image_zarr, node)
        level = 0  # load full resolution image
        numpy_type = image_node.data[level].dtype
        img_shape = image_node.data[level].shape

        # stitch full-resolution images into a grid
        def get_field(tile_name: str) -> np.ndarray:
            """tile_name is 'row,col'"""
            row, col = (int(n) for n in tile_name.split(","))
            field_index = (column_count * row) + col
            path = f"{field_index}/{level}"
            LOGGER.debug(f"LOADING tile... {path}")
            try:
                data = self.zarr.load(path)
            except ValueError:
                LOGGER.error(f"Failed to load {path}")
                data = np.zeros(img_shape, dtype=numpy_type)
            return data

        lazy_reader = delayed(get_field)

        def get_lazy_well() -> da.Array:
            lazy_rows = []
            # For level 0, return whole image for each tile
            for row in range(row_count):
                lazy_row: List[da.Array] = []
                for col in range(column_count):
                    tile_name = f"{row},{col}"
                    LOGGER.debug(f"creating lazy_reader. row:{row} col:{col}")
                    lazy_tile = da.from_delayed(
                        lazy_reader(tile_name), shape=img_shape, dtype=numpy_type
                    )
                    lazy_row.append(lazy_tile)
                lazy_rows.append(da.concatenate(lazy_row, axis=4))
            return da.concatenate(lazy_rows, axis=3)

        node.data = [get_lazy_well()]
        node.metadata = image_node.metadata


class Plate(Spec):
    @staticmethod
    def matches(zarr: ZarrLocation) -> bool:
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
        LOGGER.info("plate_data", self.plate_data)
        self.rows = self.plate_data.get("rows")
        self.columns = self.plate_data.get("columns")
        self.acquisitions = self.plate_data.get("acquisitions")
        self.first_field = "0"
        self.row_names = [row["name"] for row in self.rows]
        self.col_names = [col["name"] for col in self.columns]

        self.well_paths = [well["path"] for well in self.plate_data.get("wells")]
        self.well_paths.sort()

        self.run = ""
        # TEMP - support acquisition path in plate/acq/row/col hierarchy
        # remove when we don't want to support dev versions of ome-zarr plate data
        if len(self.acquisitions) > 0:
            self.run = self.acquisitions[0].get("path", "")
            if len(self.run) > 0 and not self.run.endswith("/"):
                self.run = self.run + "/"

        self.row_count = len(self.rows)
        self.column_count = len(self.columns)
        # Get the first image...
        path = f"{self.well_paths[0]}/{self.first_field}"
        image_zarr = self.zarr.create(path)
        image_node = Node(image_zarr, node)

        self.numpy_type = self.get_numpy_type(image_node)
        img_shape = image_node.data[0].shape

        img_pyramid_shapes = [d.shape for d in image_node.data]

        LOGGER.debug("img_pyramid_shapes", img_pyramid_shapes)

        size_y = img_shape[3]
        size_x = img_shape[4]

        # FIXME - if only returning a single stiched plate (not a pyramid)
        # need to decide optimal size. E.g. longest side < 1500
        TARGET_SIZE = 1500
        plate_width = self.column_count * size_x
        plate_height = self.row_count * size_y
        longest_side = max(plate_width, plate_height)
        target_level = 0
        for level, shape in enumerate(img_pyramid_shapes):
            plate_width = self.column_count * shape[-1]
            plate_height = self.row_count * shape[-2]
            longest_side = max(plate_width, plate_height)
            target_level = level
            if longest_side <= TARGET_SIZE:
                break

        LOGGER.debug("target_level", target_level)

        pyramid = []

        # This should create a pyramid of levels, but causes seg faults!
        # for level in range(5):
        for level in [target_level]:

            tile_shape = img_pyramid_shapes[level]
            lazy_plate = self.get_stitched_grid(level, tile_shape)
            pyramid.append(lazy_plate)

        # Set the node.data to be pyramid view of the plate
        node.data = pyramid
        # Use the first image's metadata for viewing the whole Plate
        node.metadata = image_node.metadata

        # "metadata" dict gets added to each 'plate' layer in napari
        node.metadata.update({"metadata": {"plate": self.plate_data}})

    def get_numpy_type(self, image_node: Node) -> np.dtype:
        return image_node.data[0].dtype

    def get_tile_path(self, level: int, row: int, col: int) -> str:
        return (
            f"{self.run}{self.row_names[row]}/"
            f"{self.col_names[col]}/{self.first_field}/{level}"
        )

    def get_stitched_grid(self, level: int, tile_shape: tuple) -> da.core.Array:
        def get_tile(tile_name: str) -> np.ndarray:
            """tile_name is 'level,z,c,t,row,col'"""
            row, col = (int(n) for n in tile_name.split(","))
            path = self.get_tile_path(level, row, col)
            LOGGER.debug(f"LOADING tile... {path}")

            try:
                data = self.zarr.load(path)
            except ValueError:
                LOGGER.error(f"Failed to load {path}")
                data = np.zeros(tile_shape, dtype=self.numpy_type)
            return data

        lazy_reader = delayed(get_tile)

        lazy_rows = []
        # For level 0, return whole image for each tile
        for row in range(self.row_count):
            lazy_row: List[da.Array] = []
            for col in range(self.column_count):
                tile_name = f"{row},{col}"
                lazy_tile = da.from_delayed(
                    lazy_reader(tile_name), shape=tile_shape, dtype=self.numpy_type
                )
                lazy_row.append(lazy_tile)
            lazy_rows.append(da.concatenate(lazy_row, axis=4))
        return da.concatenate(lazy_rows, axis=3)


class PlateLabels(Plate):
    def get_tile_path(self, level: int, row: int, col: int) -> str:
        """251.zarr/A/1/0/labels/0/3/"""
        path = (
            f"{self.row_names[row]}/{self.col_names[col]}/"
            f"{self.first_field}/labels/0/{level}"
        )
        return path

    def get_pyramid_lazy(self, node: Node) -> None:
        super().get_pyramid_lazy(node)
        # pyramid data may be multi-channel, but we only have 1 labels channel
        node.data[0] = node.data[0][:, 0:1, :, :, :]
        # remove image metadata
        node.metadata = {}

        # combine 'properties' from each image
        # from https://github.com/ome/ome-zarr-py/pull/61/
        properties: Dict[int, Dict[str, Any]] = {}
        for row in self.row_names:
            for col in self.col_names:
                path = f"{row}/{col}/{self.first_field}/labels/0/.zattrs"
                labels_json = self.zarr.get_json(path).get("image-label", {})
                # NB: assume that 'label_val' is unique across all images
                props_list = labels_json.get("properties", [])
                if props_list:
                    for props in props_list:
                        label_val = props["label-value"]
                        properties[label_val] = dict(props)
                        del properties[label_val]["label-value"]
        node.metadata["properties"] = properties

    def get_numpy_type(self, image_node: Node) -> np.dtype:
        # FIXME - don't assume Well A1 is valid
        path = self.get_tile_path(0, 0, 0)
        label_zarr = self.zarr.load(path)
        return label_zarr.dtype


class Reader:
    """Parses the given Zarr instance into a collection of Nodes properly ordered
    depending on context.

    Depending on the starting point, metadata may be followed up or down the Zarr
    hierarchy.
    """

    def __init__(self, zarr: ZarrLocation) -> None:
        assert zarr.exists()
        self.zarr = zarr
        self.seen: List[ZarrLocation] = [zarr]

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
