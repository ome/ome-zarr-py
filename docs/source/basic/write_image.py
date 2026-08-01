# %% [markdown]
# # Write OME-ZARR images
# (basic:write)=
#
# Writing ome-zarr images is primarily exposed through the {py:class}`ome_zarr.classes.image.OMEZarrImage` and {py:class}`ome_zarr.classes.image.OMEZarrMultiscales` classes, which provide a high-level API for creating and manipulating OME-ZARR images and pyramids.

# %%
import numpy as np

from ome_zarr import OMEZarrImage, OMEZarrMultiscale

# %% [markdown]
# Let's first create some random data to write:

# %%
# create some random data to write
size_xy = 128
size_z = 10
rng = np.random.default_rng(0)
data = rng.poisson(lam=10, size=(2, size_z, size_xy, size_xy)).astype(np.uint8)

# %% [markdown]
# We then create an {py:class}`OMEZarrImage` from our data,
# where we can specify some basic metadata for the image data, such as the types of axes (`czyx`) and their scales and units.
# The {py:class}`OMEZarrMultiscale` class creation then builds a multiscale pyramid of dask arrays by downsampling as specified by the `scale_factors` parameter.
# You can use this class to pass how viewers should render the image by specifying optional parameters such as `channel_names`, `channel_colors` and `contrast_limits`.
# As a last step, we write the multiscale image to disk using the `to_ome_zarr` method, which will create a valid OME-ZARR file that can be read by any OME-ZARR compatible viewer.
#
# ```{hint}
# The demonstrated writer method below defaults to writing OME-ZARR version `0.5`,
# but also supports writing OME-ZARR version `0.6.dev4` and `0.4`.
# ```

# %%
image = OMEZarrImage(
    data=data,
    axes=["c", "z", "y", "x"],
    scale={"c": 1.0, "z": 0.5, "y": 0.1, "x": 0.1},
    axes_units={"z": "micrometer", "y": "micrometer", "x": "micrometer"},
)

multiscales = OMEZarrMultiscale(
    image=image,
    scale_factors=(2, 4, 8),
    method="resize",
    channel_names=["DAPI", "GFP"],  # optional
    channel_colors=["00FFFF", "FF00FF"],  # optional
    contrast_limits=[(0, 255), (0, 255)],  # optional
)
multiscales.to_ome_zarr("test_ngff.ome.zarr", version="0.5")

# %%
multiscales.images

# %% [markdown]
# ## API alternative: Direct write
#
# Besides the above-described class-based approach, another principle entry-point for writing OME-ZARR images is using the {py:func}`ome_zarr.writer.write_image` function.
# This takes an n-dimensional `numpy` array or `dask` array and writes it to the specified zarr group according to the OME-ZARR specification.
# By default, a pyramid of resolution levels will be created by down-sampling the data by a factor of 2 in the X and Y dimensions.
# For more custom control over the pyramid, see the more in-depth example on [scaling functions and scale factors](advanced:pyramid).

# %%
from ome_zarr.writer import write_image

path = "test_ngff_image.ome.zarr"
write_image(data, "test_ngff2.ome.zarr", axes="czyx")

# %% [markdown]
# Alternatively, the {py:func}`ome_zarr.writer.write_multiscale` can be used,
# which takes a "pyramid" of pre-computed `numpy` arrays.
#
# The default version of OME-NGFF is v0.5, which is based on Zarr v3.
# A zarr v3 group and store is created by `zarr.open_group()` below.
# To write OME-NGFF v0.4 (Zarr v2), pass the `fmt=FormatV04()` argument.

# %%
from ome_zarr.format import FormatV04

path = "test_ngff_image_v2.ome.zarr"
write_image(data, path, axes="czyx", fmt=FormatV04())

# %% [markdown]
# To view the image, see tutorial on [viewing images](basic:view_images).
