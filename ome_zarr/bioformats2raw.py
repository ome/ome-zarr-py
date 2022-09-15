"""
Spec definitions
"""

import logging
import os
import re
import tempfile
from xml.etree import ElementTree as ET

import ome_types
from ome_zarr_metadata import __version__  # noqa

from ome_zarr.io import ZarrLocation
from ome_zarr.reader import Node
from ome_zarr.reader import Spec as Base

__author__ = "Open Microscopy Environment (OME)"
__copyright__ = "Open Microscopy Environment (OME)"
__license__ = "BSD-2-Clause"

_logger = logging.getLogger(__name__)


class bioformats2raw(Base):
    @staticmethod
    def matches(zarr: ZarrLocation) -> bool:
        layout = zarr.root_attrs.get("bioformats2raw.layout", None)
        return layout == 3

    def __init__(self, node: Node) -> None:
        super().__init__(node)
        try:
            data = self.handle(node)
            if data.plates:
                _logger.info("Plates detected. Skipping implicit loading")
            else:
                for idx, image in enumerate(data.images):
                    series = node.zarr.create(str(idx))
                    assert series.exists(), f"{series} is missing"
                    _logger.info("found %s", series)
                    subnode = node.add(series)
                    if subnode:
                        subnode.metadata["ome-xml:index"] = idx
                        subnode.metadata["ome-xml:image"] = image
            node.metadata["ome-xml"] = data
        except Exception:
            _logger.exception("failed to parse metadata")

    def fix_xml(self, ns: str, elem: ET.Element) -> None:
        """
        Note: elem.insert() was not updating the object correctly.
        """

        if elem.tag == f"{ns}Pixels":

            must_have = {f"{ns}BinData", f"{ns}TiffData", f"{ns}MetadataOnly"}
            children = {x.tag for x in elem}

            if not any(x in children for x in must_have):
                # Needs fixing
                metadata_only = ET.Element(f"{ns}MetadataOnly")

                last_channel = -1
                for idx, child in enumerate(elem):
                    if child.tag == f"{ns}Channel":
                        last_channel = idx
                elem.insert(last_channel + 1, metadata_only)

        elif elem.tag == f"{ns}Plane":
            remove = None
            for idx, child in enumerate(elem):
                if child.tag == f"{ns}HashSHA1":
                    remove = child
            if remove:
                elem.remove(remove)

    def parse_xml(self, filename: str) -> ome_types.model.OME:
        # Parse the file and find the current schema
        root = ET.parse(filename)
        m = re.match(r"\{.*\}", root.getroot().tag)
        ns = m.group(0) if m else ""

        # Update the XML to include MetadataOnly
        for child in list(root.iter()):
            self.fix_xml(ns, child)
        fixed = ET.tostring(root.getroot()).decode()

        # Write file out for ome_types
        with tempfile.NamedTemporaryFile() as t:
            t.write(fixed.encode())
            t.flush()
            return ome_types.from_xml(t.name)

    def handle(self, node: Node) -> ome_types.model.OME:
        metadata = node.zarr.subpath("OME/METADATA.ome.xml")
        _logger.info("Looking for metadata in %s", metadata)
        if os.path.exists(metadata):
            return self.parse_xml(metadata)
