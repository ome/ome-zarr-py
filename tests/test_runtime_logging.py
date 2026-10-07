"""Runtime notices should go through logging, not the warnings module."""

import logging

from ome_zarr.axes import Axes


def test_unknown_axis_unit_is_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="ome_zarr.axes"):
        Axes(["x"], axes_units={"x": "furlong"})

    assert any(
        "not a known spatial unit" in record.message and "furlong" in record.message
        for record in caplog.records
    )
