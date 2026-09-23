import numpy as np
from ome_zarr_models.v06.coordinate_transforms import (
    Axis,
    CoordinateSystem,
    CoordinateSystemIdentifier,
)

from ome_zarr.classes.image import OMEZarrImage, OMEZarrMultiscale


def test_get_local_coordinate_system():
    cs1 = CoordinateSystem(
        name="cs1", axes=(Axis(name="y", type="space"), Axis(name="x", type="space"))
    )
    cs2 = CoordinateSystem(
        name="cs2", axes=(Axis(name="x", type="space"), Axis(name="y", type="space"))
    )
    img = OMEZarrMultiscale(
        OMEZarrImage(np.zeros((32, 64), float), axes=["y", "x"]),
        coordinate_systems=[cs1, cs2],
    )
    assert img.get_local_coordinate_system("cs2") == cs2
    assert img.get_local_coordinate_system("notacoordinatesystem") is None


def test_get_coordinate_system():
    cs1 = CoordinateSystem(
        name="cs1", axes=(Axis(name="y", type="space"), Axis(name="x", type="space"))
    )
    cs2 = CoordinateSystem(
        name="cs2", axes=(Axis(name="x", type="space"), Axis(name="y", type="space"))
    )
    img = OMEZarrMultiscale(
        OMEZarrImage(np.zeros((32, 64), float), axes=["y", "x"]),
        coordinate_systems=[cs1, cs2],
    )
    assert img.get_coordinate_system(CoordinateSystemIdentifier(name="cs2")) == cs2
    assert (
        img.get_coordinate_system(
            CoordinateSystemIdentifier(name="notacoordinatesystem")
        )
        is None
    )


# TODO: test getting coordinate system from a child label multiscale.
# For now, there is no way of creating a label multiscale with coordinate systems.
