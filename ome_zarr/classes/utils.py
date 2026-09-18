from ome_zarr_models.v06.coordinate_transforms import (
    AnyTransform,
    CoordinateSystem,
)

import transformnd as tnd
import posixpath
from ome_zarr.classes.image import OMEZarrMultiscale

def _ozmp_tf_to_tnd(
    transform: AnyTransform,
    zarr_context: str = "",
    source_cs: CoordinateSystem | None = None,
    target_cs: CoordinateSystem | None = None,
    coordinate_displacements: dict[str, OMEZarrMultiscale] | None = None,
) -> tnd.base.Transform:
    """
    Convert an OME-Zarr coordinate transformation to a transformnd Transform object.

    Parameters:
    -----------
    transform: ome_zarr_models.v06.coordinate_transforms.AnyTransform
        OME-Zarr coordinate transformation to convert.
    zarr_context: str, optional
        Context path within the Zarr store, used to resolve relative paths.
    source_cs: ome_zarr_models.v06.coordinate_transforms.CoordinateSystem | None, optional
        Source coordinate system for the transformation.
    target_cs: ome_zarr_models.v06.coordinate_transforms.CoordinateSystem | None, optional
        Target coordinate system for the transformation.
    coordinate_displacements: dict[str, ome_zarr.classes.image.OMEZarrMultiscale] | None, optional
        Dictionary of coordinate displacement fields keyed by their paths.

    """
    import numpy as np

    if transform.input is not None:
        input_path = transform.input.path or ""
        output_path = transform.output.path or ""

        # zarr_context prepends path with relative path from root
        # to keep track of global location of coordinate systems in the zarr store
        if zarr_context:
            input_path = (
                posixpath.join(zarr_context, input_path)
                if input_path
                else zarr_context
            )
            output_path = (
                posixpath.join(zarr_context, output_path)
                if output_path
                else zarr_context
            )

        spaces = tnd.Spaces(
            (input_path, transform.input.name),
            (output_path, transform.output.name),
        )
    else:
        spaces = tnd.Spaces(None, None)

    tnd_transform = None
    # Example for an affine transformation (this will depend on the actual structure of AnyTransform)
    if transform.type == "affine":
        aff = np.asarray(transform.affine)
        if aff.shape[0] == aff.shape[1]:
            tnd_transform = tnd.transforms.Affine(
                transform.affine,
                spaces=spaces,
            )
        else:
            aff = np.eye(max(aff.shape))
            aff[: aff.shape[0], : aff.shape[1]] = aff
            tnd_transform = tnd.transforms.Affine(aff, spaces=spaces)

    elif transform.type == "displacements":
        path_to_dfield = transform.path or ""
        if zarr_context and path_to_dfield:
            path_to_dfield = posixpath.join(zarr_context, path_to_dfield)

        if coordinate_displacements is not None:
            dfield = coordinate_displacements.get(
                posixpath.basename(path_to_dfield)
            )
            if dfield is not None:
                if dfield.images[0].scale is None:
                    raise ValueError(
                        f"Displacement field at {path_to_dfield} is missing scale information."
                    )
                tnd_transform = tnd.transforms.Displacements(
                    dfield.images[0].data,
                    index_transform=tnd.transforms.Scale(
                        list(dfield.images[0].scale.values())[1:]
                    ),
                    vector_axis=0,
                    spaces=spaces,
                )
    elif transform.type == "mapAxis":
        tnd_transform = tnd.transforms.MapAxis(
            list(transform.mapAxis),
            spaces=spaces,
        )

    elif transform.type == "projectAxis":
        tnd_transform = tnd.transforms.ProjectAxis(
            created=transform.createdOutputs,
            dropped=transform.droppedInputs,
            spaces=spaces,
            source_ndim=len(source_cs.axes) if source_cs is not None else None,
            target_ndim=len(target_cs.axes) if target_cs is not None else None,
        )

    elif transform.type == "scale":
        tnd_transform = tnd.transforms.Scale(transform.scale, spaces=spaces)

    elif transform.type == "translation":
        tnd_transform = tnd.transforms.Translate(
            transform.translation, spaces=spaces
        )

    elif transform.type == "rotation":
        affine_matrix = np.eye(len(transform.rotation) + 1)
        affine_matrix[:-1, :-1] = transform.rotation
        tnd_transform = tnd.transforms.Affine(affine_matrix, spaces=spaces)

    elif transform.type == "byDimension":
        sub_transformations = transform.transformations
        tnd_sub_transforms = [
            tnd.transforms.by_dimension.SubTransform(
                transform=_ozmp_tf_to_tnd(sub_tf.transformation),
                input_axes=sub_tf.inputAxes,
                output_axes=sub_tf.outputAxes,
            )
            for sub_tf in sub_transformations
        ]
        tnd_transform = tnd.transforms.ByDimension(
            subtransforms=tnd_sub_transforms,
            fill_identity=0,
            spaces=spaces,
        )
    elif transform.type == "sequence":
        sub_transformations = transform.transformations
        tnd_sub_transforms = [
            _ozmp_tf_to_tnd(sub_tf, zarr_context, source_cs, target_cs)
            for sub_tf in sub_transformations
        ]
        tnd_transform = tnd.base.TransformSequence(
            tnd_sub_transforms,
            spaces=spaces,
        )

    return tnd_transform