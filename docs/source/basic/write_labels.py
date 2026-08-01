# %% [markdown]
# # Write labels
# (basic:labels)=
# 
# Storing labels data alongside image data is a key application and feature of the ome-zarr file standard.
# First, let's create some image data:

# %%
import numpy as np
from skimage.data import binary_blobs

from ome_zarr import OMEZarrImage, OMEZarrLabels, OMEZarrMultiscale

# %%
mean_val = 10
size = 64
rng = np.random.default_rng(0)
data = rng.poisson(mean_val, size=(size, size, size)).astype(np.uint8)

# %% [markdown]
# We can directly turn this into an {py:class}`ome_zarr.classes.image.OMEZarrMultiscales` object:

# %%
ngff_image = OMEZarrImage(
    data=data,
    axes="zyx",
)
ngff_multiscales = OMEZarrMultiscale(image=ngff_image)

# %% [markdown]
# In a next step, we create some dummy labels data:

# %%
# add labels...
blobs = binary_blobs(length=size, volume_fraction=0.1, n_dim=3).astype('int8')
blobs2 = binary_blobs(length=size, volume_fraction=0.1, n_dim=3).astype('int8')
# blobs will contain values of 1, 2 and 0 (background)
blobs += 2 * blobs2

# %% [markdown]
# We now turn these two label images into instances of {py:class}`ome_zarr.classes.image.OMEZarrLabels` similar to how we did above for the image data:

# %%
label_ngff1 = OMEZarrImage(
    data=blobs,
    axes="zyx",
)
label_ngff2 = OMEZarrImage(
    data=blobs2,
    axes="zyx",
)

# create OMEZarrMultiscales for labels
labels_multiscales1 = OMEZarrLabels(image=label_ngff1)
labels_multiscales2 = OMEZarrLabels(image=label_ngff2)

# %% [markdown]
# We can now add the labels as an attribute of the image data and write the whole thing to disk:

# %%
ngff_multiscales.labels = {
    "labels1": labels_multiscales1,
    "labels2": labels_multiscales2
}

ngff_multiscales.to_ome_zarr("ngff_multiscales_with_labels.zarr", overwrite=True)

# %% [markdown]
# This automatically creates the following file structure on disk:
# 
# ```
# ngff_multiscales_with_labels.zarr
# ├── zarr.json
# ├── labels
# │   ├── zarr.json
# │   ├── labels1
# |   |   ├── .zarr.json
# |   |   ├── s0
# |   |   └── s1
# │   └── labels2
# |       ├── .zarr.json
# |       ├── s0
# |       └── s1
# ├── s0
# └── s1
# ```

# %%



