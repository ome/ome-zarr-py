"""Tests for converting ome-zarr-models-py coordinate transforms to transformnd."""

import numpy as np
import pytest
from ome_zarr_models.v06.coordinate_transforms import (
    Axis,
    AnyTransform,
    CoordinateSystem,
)
from pydantic import TypeAdapter

from ome_zarr.classes.utils import _ozmp_tf_to_tnd

transform_adapter = TypeAdapter(AnyTransform)

# (transform dict, input point, expected output point, extra _ozmp_tf_to_tnd kwargs)
CASES = [
    (
        {"type": "translation", "translation": (1.0, -2.0, 3.4)},
        (0.0, 0.0, 0.0),
        (1.0, -2.0, 3.4),
        {},
    ),
    (
        {"type": "scale", "scale": (2.0, 3.0)},
        (1.0, 2.0),
        (2.0, 6.0),
        {},
    ),
    (
        {"type": "rotation", "rotation": ((0.0, -1.0), (1.0, 0.0))},
        (1.0, 0.0),
        (0.0, 1.0),
        {},
    ),
    (
        # NGFF affines are stored inhomogeneous (M x N+1); this is the shape
        # that previously hit the padding bug in _ozmp_tf_to_tnd.
        {"type": "affine", "affine": ((2.0, 0.0, 1.0), (0.0, 3.0, 2.0))},
        (1.0, 1.0),
        (3.0, 5.0),
        {},
    ),
    (
        {"type": "mapAxis", "mapAxis": (1, 2, 0, 3)},
        (11.0, 12.0, 22.0, 33.0),
        (12.0, 22.0, 11.0, 33.0),
        {},
    ),
    (
        {
            "type": "sequence",
            "transformations": [
                {"type": "scale", "scale": (2.0, 2.0)},
                {"type": "translation", "translation": (1.0, 1.0)},
            ],
        },
        (1.0, 1.0),
        (3.0, 3.0),
        {},
    ),
    (
        # inserts a new (zero) axis at position 0; needs source_cs since
        # ProjectAxis can't infer ndim from the transform alone.
        {"type": "projectAxis", "createdOutputs": (0,)},
        (5.0, 7.0),
        (0.0, 5.0, 7.0),
        {
            "source_cs": CoordinateSystem(
                name="in", axes=(Axis(name="y"), Axis(name="x"))
            )
        },
    ),
    (
        # drops the first input axis (e.g. "z")
        {"type": "projectAxis", "droppedInputs": (0,)},
        (1.0, 5.0, 7.0),
        (5.0, 7.0),
        {
            "source_cs": CoordinateSystem(
                name="in", axes=(Axis(name="z"), Axis(name="y"), Axis(name="x"))
            )
        },
    ),
]


@pytest.mark.parametrize(("transform_dict", "point", "expected", "kwargs"), CASES)
def test_ozmp_tf_to_tnd_apply(transform_dict, point, expected, kwargs):
    transform = transform_adapter.validate_python(transform_dict)
    tnd_transform = _ozmp_tf_to_tnd(transform, **kwargs)
    result = tnd_transform.apply(np.asarray([point]))
    assert result[0] == pytest.approx(expected)
