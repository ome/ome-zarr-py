# %% [markdown]
# # Customizing the pyramid
# (advanced:pyramid)=
# 
# 
# Multi-resolution pyramids are an integral part of ome-zarr image data
# and enable fast rendering of large images.
# The entrypoints to writing ome-zarr images in ome-zarr-py ({py:func}`ome_zarr.classes.image.OMEZarrMultiscale`, {py:func}`ome_zarr.writer.write_image` and {py:func}`ome_zarr.writer.write_labels`)
# build these pyramids under the hood as delayed dask arrays based on the settings for the scaling functions and scale factors.
# 
# In this example, the downsampling will be applied in all spatial dimensions *except the z dimension*, which will be left at a scale factor of 1.
# To apply equal or custom downsampling factors along all spatial dimensions, pass the scale factors as a list of dicts (see [below](#advanced:custom-downsampling-values)).

# %%
import numpy as np

from ome_zarr import OMEZarrImage, OMEZarrLabels, OMEZarrMultiscale
from ome_zarr.writer import write_image

scale_factors = [2, 4, 8]
rng = np.random.default_rng(0)
data = rng.poisson(lam=10, size=(64, 64, 64)).astype(np.uint8)

# %%
write_image(
    data,
    "test_ngff_image.ome.zarr",
    axes="zyx",
    scale_factors=scale_factors,
)

# %% [markdown]
# Or, if you choose to work with the {py:class}`ome_zarr.classes.image.OMEZarrMultiscale` and {py:class}`ome_zarr.classes.image.OMEZarrImage` classes directly,
# they accept the same arguments:

# %%
ngff_image = OMEZarrImage(data, axes="zyx")
ngff_multiscales = OMEZarrMultiscale(ngff_image, scale_factors=scale_factors)

ngff_multiscales.to_ome_zarr("test_ngff_image_multiscales.ome.zarr")

# %% [markdown]
# ## Custom downsampling values
# (advanced:custom-downsampling-values)=
# 
# To specify custom downsampling values, pass a list of dictionaries with the keys being the names of the axes to the writer function like in the following example.
# This will apply equal downsampling factors along all present axes (`zyx` in this case):

# %%
scale_factors = [
    {"z": 2,"x": 2, "y": 2},
    {"z": 4,"x": 4, "y": 4},
    {"z": 8,"x": 8, "y": 8},
]

write_image(
    data,
    "test_ngff_image_custom_scale.ome.zarr",
    axes="zyx",
    scale_factors=scale_factors,
    )

# %% [markdown]
# ## Custom downsampling functions
# 
# ome-zarr-py provides multiple methods for downsampling, which can be found in the {py:class}`ome_zarr.scale.Methods` class:

# %%
from ome_zarr.scale import Methods

print([m.value for m in Methods])

# %% [markdown]
# You can use one of these functions for downsampling by passing the method name as a string to the writer function, e.g. `method="local_mean"` or `method="resize"`, i.e.:

# %%
write_image(
    data,
    "test_ngff_image_custom_method.ome.zarr",
    axes="zyx",
    scale_factors=scale_factors,
    method="nearest"
)


# %% [markdown]
# Again, the {py:class}`ome_zarr.OMEZarrMultiscale` accepts the same argument:

# %%
ngff_image = OMEZarrImage(data, axes="zyx")
ngff_multiscales = OMEZarrMultiscale(ngff_image, scale_factors=scale_factors, method="nearest")

ngff_multiscales.to_ome_zarr("test_ngff_image_multiscales.ome.zarr")

# %% [markdown]
# ```{warning}
# 
# The choice of the correct downsampling function is typically of secondary importance,
# *unless* your data specifically requires a certain method.
# 
# For instance, when writing categorical data (i.e., segmentations or generally labels),
# you will want to use a method that preserves the label values, such as {py:func}`ome_zarr.scale.Methods.NEAREST`.
# 
# See also section on [writing labels](basic:labels)
# ```
# 
# If you specifically want to create a labels image, you can use the {py:class}`ome_zarr.classes.image.OMEZarrLabels` class.
# It provides the same functionality as the {py:class}`ome_zarr.classes.image.OMEZarrMultiscale` class, but the downsampling function used under the hood defaults to {py:func}`ome_zarr.scale.Methods.NEAREST`,
# which is typically the most appropriate for labels data.
# 
# For more information on writing labels data, see the [labels writing section](advanced/labels/index).

# %%
ngff_image = OMEZarrImage(data, axes="zyx")
ngff_multiscales = OMEZarrLabels(ngff_image, scale_factors=scale_factors)

ngff_multiscales.to_ome_zarr("test_ngff_image_multiscales_labels.ome.zarr")


