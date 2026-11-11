# %% [markdown]
# # Bring your own pyramid
# (advanced:bring_own_pyramid)=
#
#
# In some scenarios, you ma already have a pre-computed multi-resolution pyramid and do not want to rely on the downsampling implemented in ome-zarr-py.
# This can be the case if you:
# - Want to use a particular downsampling algorithm that's not part of ome-zarr-py
# - Already have the pyramid generated from another source and want to avoid recomputation
#
# In these cases, you can construct the {py:func}`ome_zarr.classes.image.OMEZarrMultiscale` class directly from your pre-computed pyramid.

# %%
import numpy as np
from ome_zarr.classes.image import OMEZarrImage, OMEZarrMultiscale

level0 = np.random.rand(256, 256)  # Example 2D image with shape (y, x)
level1 = level0[::2, ::2]
level2 = level1[::2, ::2]

# %% [markdown]
# We now convert these into instances of {py:func}`ome_zarr.classes.image.OMEZarrImage`:

# %%
oz_img_lvl0 = OMEZarrImage(level0, axes="yx", scale={"y": 1.0, "x": 1.0})
oz_img_lvl1 = OMEZarrImage(level1, axes="yx", scale={"y": 2.0, "x": 2.0})
oz_img_lvl2 = OMEZarrImage(level2, axes="yx", scale={"y": 4.0, "x": 4.0})

# %% [markdown]
# Now we can construct the OMEZarrMultiscale object from these images.

# %%
oz_multiscale = OMEZarrMultiscale.from_pyramid(images=[oz_img_lvl0, oz_img_lvl1, oz_img_lvl2])
