# %% [markdown]
# # Transforms with changing dimensionality
#
# A particular and important kind of transforms,
# are transforms that change the dimensionality of the data.
# Common examples for this case are:
# - 2D to 3D transforms, e.g. for aligning a 2D slice to a 3D volume
# - 2D + channel to 2D transforms, e.g. for aligning a 2D slice with multiple channels to a 2D slice with a single channel
#
# The transform that expresses this change in dimensionality is the [`ProjectAxis` transform](https://ngff.openmicroscopy.org/specifications/dev/index.html#projectaxis).
#
# This tutorial demonstrates its usage for the case of a 2D to 3D transform,
# where a 2D slice is aligned to a 3D volume.

# %%
from ome_zarr_models.v06.coordinate_transforms import Sequence
from skimage import data

from ome_zarr import OMEZarrImage, OMEZarrMultiscale, OMEZarrScene

# %%
img = data.cells3d().transpose((1, 0, 2, 3))
img.shape

# %%
some_slice = img[0, 30, :, :]
some_slice.shape

# %%
ngff_img = OMEZarrImage(
    data=img,
    axes=["c", "z", "y", "x"],
    scale={"c": 1, "z": 1, "y": 1, "x": 1},
    name="cells3d"
)

ngff_ms = OMEZarrMultiscale(
    image=ngff_img,
)

slice_img = OMEZarrImage(
    data=some_slice,
    axes=["y", "x"],
    scale={"y": 1, "x": 1},
    name="cells3d_slice"
)

slice_ms = OMEZarrMultiscale(
    image=slice_img,
)

# %%
transform_to_3d = Sequence.model_validate({
    "type": "sequence",
    "input": {"path": "cells3d_slice", "name": "physical"},
    "output": {"path": "cells3d", "name": "physical"},
    "transformations": [
        {
            "type": "projectAxis",
            "createdOutputs": [0, 1]
        },
        {
            "type": "translation",
            "translation": [0, 30, 0, 0]
        }
    ]
})

# %%
scene = OMEZarrScene(
    images=[ngff_ms, slice_ms],
    coordinate_transformations=[transform_to_3d]
)

# %%
scene.to_ome_zarr("scene_2d_to_3d.zarr", overwrite=True)
