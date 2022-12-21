"""Spec extension for reading bioformats2raw.layout.

This specification detects and reads filesets which were created by
bioformats2raw and therefore can have multiple multiscale image groups
present. Each such image will be returned by the [ome_zarr.reader.Reader]
as a separate [ome_zarr.reader.Node], but metadata which has been parsed
from the OME-XML metadata associated with the specific image will be
attached.

TBD: Example
"""

import logging
import os
import re
import tempfile
from xml.etree import ElementTree as ET

import ome_types

from ome_zarr.io import ZarrLocation
from ome_zarr.reader import Node
from ome_zarr.reader import Spec as Base

__author__ = "Open Microscopy Environment (OME)"
__copyright__ = "Open Microscopy Environment (OME)"
__license__ = "BSD-2-Clause"

_logger = logging.getLogger(__name__)


class bioformats2raw(Base):
    """A spec-type for reading multi-image filesets OME-XML
    metadata.
    """

    @staticmethod
    def matches(zarr: ZarrLocation) -> bool:
        """Pass if the metadata for the zgroup contains
        `{"bioformats2raw.layout": 3}`"""
        layout = zarr.root_attrs.get("bioformats2raw.layout", None)
        _logger.error(f"layout={layout == 3} zarr={zarr}")
        return layout == 3

    def __init__(self, node: Node) -> None:
        """Load metadata from the three sources associated with this
        specification: the OME zgroup metadata, the OME-XML file, and
        the images zgroups themselves.
        """
        super().__init__(node)
        try:

            # Load OME/METADATA.ome.xml
            data = self._handle(node)
            if data.plates:
                _logger.info("Plates detected. Skipping implicit loading")
            else:

                # Load the OME/ zgroup metadata
                ome = node.zarr.create("OME")
                if ome.exists:
                    series_metadata = ome.zarr.root_attrs.get("series", None)
                    if series_metadata is not None:
                        node.metadata["series"] = series_metadata

                # Load each individual image
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

    def _fix_xml(self, ns: str, elem: ET.Element) -> None:
        """Correct invalid OME-XML.

        Some versions of bioformats2raw did not include a MetadataOnly
        tag.

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

    def _parse_xml(self, filename: str) -> ome_types.model.OME:
        """Generate [ome_types.model.OME] from OME-XML"""
        # Parse the file and find the current schema
        root = ET.parse(filename)
        m = re.match(r"\{.*\}", root.getroot().tag)
        ns = m.group(0) if m else ""

        # Update the XML to include MetadataOnly
        for child in list(root.iter()):
            self._fix_xml(ns, child)
        fixed = ET.tostring(root.getroot()).decode()

        # Write file out for ome_types
        with tempfile.NamedTemporaryFile() as t:
            t.write(fixed.encode())
            t.flush()
            return ome_types.from_xml(t.name)

    def _handle(self, node: Node) -> ome_types.model.OME:
        """Main parsing method which looks for OME/METADATA.ome.xml"""
        metadata = node.zarr.subpath("OME/METADATA.ome.xml")
        _logger.info("Looking for metadata in %s", metadata)
        if os.path.exists(metadata):
            return self._parse_xml(metadata)
