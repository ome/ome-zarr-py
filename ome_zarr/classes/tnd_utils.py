"""Utilities for creating a transformnd.TransformGraph from OME-Zarr metadata."""

from __future__ import annotations

import posixpath
from collections.abc import Sequence
from typing import TYPE_CHECKING

import ome_zarr_models.v06.coordinate_transforms as ozmt
import transformnd as tnd

if TYPE_CHECKING:
    import dask.array as da
    import numpy as np

    from .image import OMEZarrMultiscale


VECTOR_FIELD_AXIS = 0


def ozmp_tf_to_tnd_spaced(
    transform: ozmt.AnyTransform,
    zarr_context: str = "",
    source_cs: ozmt.CoordinateSystem | None = None,
    target_cs: ozmt.CoordinateSystem | None = None,
    coordinate_displacements: dict[str, OMEZarrMultiscale] | None = None,
) -> tnd.Spaced:
    """
    Convert an OME-Zarr coordinate transformation with coordinate system information to a transformnd Spaced object.

    Raises:
        UnsupportedTransformation
            Transformation cannot be recreated in transformnd, or is lacking coordinate system information.
    """
    if coordinate_displacements is None:
        coordinate_displacements = dict()
    if transform.input is None or transform.output is None:
        raise UnsupportedTransformation("Missing coordinate system", transform)

    t = _ozmp_tf_to_tnd(
        transform,
        zarr_context,
        _get_ndim(source_cs),
        _get_ndim(target_cs),
        coordinate_displacements,
    )

    input_path = transform.input.path or ""
    output_path = transform.output.path or ""

    # zarr_context prepends path with relative path from root
    # to keep track of global location of coordinate systems in the zarr store
    if zarr_context:
        input_path = (
            posixpath.join(zarr_context, input_path) if input_path else zarr_context
        )
        output_path = (
            posixpath.join(zarr_context, output_path) if output_path else zarr_context
        )

    spaces = (
        (input_path, transform.input.name),
        (output_path, transform.output.name),
    )

    return tnd.Spaced(t, *spaces)


def _setup_vectorfield_args(
    transform: ozmt.Displacements | ozmt.Coordinates,
    zarr_context: str,
    coordinate_displacements: dict[str, OMEZarrMultiscale],
) -> tuple[da.Array | np.ndarray, tnd.Transform, int]:
    """Set up the arguments for vector field transformations.

    i.e. displacements, coordinates.

    Returns:
        dask.array.Array or numpy.ndarray
            Vector field array
        transformnd.Transform
            index_transformation argument
        int
            vector_axis argument

    Raises:
        UnsupportedTransformation
            If vector field is missing or malformed
    """
    path_to_vfield = transform.path or ""
    if zarr_context and path_to_vfield:
        path_to_vfield = posixpath.join(zarr_context, path_to_vfield)

    vfield = coordinate_displacements.get(posixpath.basename(path_to_vfield))
    if vfield is None:
        raise UnsupportedTransformation("Missing vector field", transform)

    img = vfield.images[0]
    if img.scale is None:
        raise UnsupportedTransformation(
            "Vector field missing scale information", transform
        )

    # Metadata is defined from displacements to input space,
    # but transformnd expects input space to displacements,
    # so this is inverted.
    index_transform = ~tnd.transforms.Scale(list(img.scale.values())[1:])

    return (
        img.data,
        index_transform,
        VECTOR_FIELD_AXIS,
    )


def _ozmp_tf_to_tnd(
    transform: ozmt.AnyTransform,
    zarr_context: str,
    source_ndim: int | None,
    target_ndim: int | None,
    coordinate_displacements: dict[str, OMEZarrMultiscale],
) -> tnd.Transform:
    """
    Convert an OME-Zarr coordinate transformation to a transformnd Transform object.

    Returns:
        transformnd.Transform
            If it can be constructed, None otherwise.

    Raises:
        UnsupportedTransformation
            Transformation cannot be recreated in transformnd.
    """
    import numpy as np

    # Example for an affine transformation (this will depend on the actual structure of AnyTransform)
    if isinstance(transform, ozmt.Affine):
        try:
            aff = np.asarray(transform.affine_matrix)
        except NotImplementedError as e:
            raise UnsupportedTransformation(
                "Path-form transformations are not implemented", transform
            ) from e

        return tnd.transforms.Affine.from_linear_map(aff[:, :-1], aff[:, -1])

    elif isinstance(transform, ozmt.Coordinates):
        arr, index_transform, vector_axis = _setup_vectorfield_args(
            transform,
            zarr_context,
            coordinate_displacements,
        )
        return tnd.transforms.Coordinates(
            arr,
            index_transform=index_transform,
            vector_axis=vector_axis,
        )

    elif isinstance(transform, ozmt.Displacements):
        arr, index_transform, vector_axis = _setup_vectorfield_args(
            transform,
            zarr_context,
            coordinate_displacements,
        )
        return tnd.transforms.Displacements(
            arr,
            index_transform=index_transform,
            vector_axis=vector_axis,
        )

    elif isinstance(transform, ozmt.Identity):
        ndim = source_ndim or target_ndim
        if ndim is None:
            raise UnsupportedTransformation("Could not infer dimensionality", transform)
        return tnd.transforms.Identity(ndim)

    elif isinstance(transform, ozmt.MapAxis):
        return tnd.transforms.MapAxis(
            list(transform.mapAxis),
        )

    elif isinstance(transform, ozmt.ProjectAxis):
        if source_ndim is None and target_ndim is None:
            raise UnsupportedTransformation("Could not infer dimensionality", transform)
        return tnd.transforms.ProjectAxis(
            created=_set_or_none(transform.createdOutputs),
            dropped=_set_or_none(transform.droppedInputs),
            source_ndim=source_ndim,
            target_ndim=target_ndim,
        )

    elif isinstance(transform, ozmt.Scale):
        return tnd.transforms.Scale(transform.scale)

    elif isinstance(transform, ozmt.Translation):
        return tnd.transforms.Translate(
            transform.translation,
        )

    elif isinstance(transform, ozmt.Rotation):
        try:
            rot = transform.rotation_matrix
        except NotImplementedError as e:
            raise UnsupportedTransformation(
                "Path-form transformations are not yet implemented", transform
            ) from e
        return tnd.transforms.Affine.from_linear_map(rot)

    elif isinstance(transform, ozmt.ByDimension):
        tnd_sub_transforms = [
            tnd.transforms.SubTransform(
                transform=_ozmp_tf_to_tnd(
                    t.transformation,
                    zarr_context,
                    len(t.inputAxes),
                    len(t.outputAxes),
                    coordinate_displacements,
                ),
                input_axes=list(t.inputAxes),
                output_axes=list(t.outputAxes),
            )
            for t in transform.transformations
        ]

        return tnd.transforms.ByDimension(tnd_sub_transforms)

    elif isinstance(transform, ozmt.Sequence):
        ts = transform.transformations
        if not transform.transformations:
            ndim = source_ndim or target_ndim
            if ndim is None:
                raise UnsupportedTransformation(
                    "Could not infer dimensionality", transform
                )
            return tnd.TransformSequence.empty(ndim)

        elif len(transform.transformations) == 1:
            return tnd.TransformSequence(
                [
                    _ozmp_tf_to_tnd(
                        ts[0],
                        zarr_context,
                        source_ndim,
                        target_ndim,
                        coordinate_displacements,
                    )
                ]
            )

        inner = [
            _ozmp_tf_to_tnd(
                ts[0], zarr_context, source_ndim, None, coordinate_displacements
            )
        ]
        for t in ts[1:-1]:
            src_ndim = inner[-1].ndims.target
            inner.append(
                _ozmp_tf_to_tnd(
                    t, zarr_context, src_ndim, None, coordinate_displacements
                )
            )
        src_ndim = inner[-1].ndims.target
        inner.append(
            _ozmp_tf_to_tnd(
                ts[-1], zarr_context, src_ndim, target_ndim, coordinate_displacements
            )
        )

        return tnd.TransformSequence(inner)

    raise UnsupportedTransformation("Unsupported transform type", transform)


def _set_or_none(it: Sequence[int] | None) -> set[int] | None:
    if it is None:
        return None
    return set(it)


def _get_ndim(cs: ozmt.CoordinateSystem | None) -> int | None:
    if cs is None:
        return None
    return len(cs.axes)


class UnsupportedTransformation(Exception):
    """Exception where ome-zarr-models has deserialised transformation metadata
    but it could not be converted into a "functional" form."""

    def __init__(self, msg: str, parsed: ozmt.Transform) -> None:
        self.msg = msg
        self.parsed = parsed

    def __str__(self) -> str:
        return f"{self.msg}: {self.parsed}"
