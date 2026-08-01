# %% [markdown]
# # Adding labels to ome-zarrs
# (advanced:labels:adding_labels)=
# 
# Unlike demonstrated in the [basic labels example](basic:labels), in many scenarios, it is not desirable to write a complete structure consisting of an ome-zarr image and some labels at once.
# Instead, one might want to *first* write the ome-zarr image (i.e., after image conversion), and *then* add the labels later (i.e., after segmentation).
# 
# The ome-zarr-py allows to do this in a lean and top-level way.

# %%
import numpy as np
from skimage.data import binary_blobs

from ome_zarr import OMEZarrImage, OMEZarrLabels, OMEZarrMultiscale

# %% [markdown]
# First, let's create some sample ome-zarr image data and write it to disk:

# %%
rng = np.random.default_rng(0)
data = rng.poisson(10, size=(64, 64, 64)).astype(np.uint8)

ngff_image = OMEZarrImage(data, axes="zyx", name="image")
ngff_multiscales = OMEZarrMultiscale(image=ngff_image)

# write the image data to disk
ngff_multiscales.to_ome_zarr("image_with_labels.zarr", overwrite=True)

# %% [markdown]
# ## Adding labels data
# 
# Now, we perform a pseudo-segmentation and create some labels data.
# The created data is then also converted into an instance of {py:class}`ome_zarr.classes.NgffMultiscales`.

# %%
labels = OMEZarrImage(
    data=binary_blobs(length=64, volume_fraction=0.1, n_dim=3).astype('int8'),
    axes="zyx",
)
labels_multiscales = OMEZarrLabels(image=labels)

# %% [markdown]
# If we assume that the image data has been written to disk previously, we need to open the existing ome-zarr file to append our labels data to the `labels` attribute of the parent image.
# Since no label images have yet been added to the group, we find the `labels` attribute to be empty:

# %%
parent_image = OMEZarrMultiscale.from_ome_zarr("image_with_labels.zarr")
print(parent_image.labels)

# %% [markdown]
#  This attribute is going to be a Python dictionary, so new images can be added to it as follows:

# %%
parent_image.labels = {
    "label_image_1": labels_multiscales
  }

# %% [markdown]
# We can now store the additional labels data to the already existing group under `"image_with_labels.zarr/labels/label_image_1"`. The name of the label image group corresponds to the key of the dictionary entry above.
# 
# ```{note}
# When writing the labels data, we want to **append** the labels data, not write the entire image data again. For this to happen, we need to set the `overwrite` argument of the `to_ome_zarr` method to `False`
# ```
# 
# 

# %%
parent_image.to_ome_zarr("image_with_labels.zarr", overwrite=False)

# %% [markdown]
# ## Appending more labels data
# 
# In the above example, we added a single label image to the parent image.
# However, it is possible to distinctively append labels images rather than adding them altogether. If we open the ome-zarr file again, we can find the already added label image under the `labels` attribute of the parent image:

# %%
parent_image = OMEZarrMultiscale.from_ome_zarr("image_with_labels.zarr")
parent_image.labels

# %% [markdown]
# We can now append more label image to the `labels` attribute of the parent image...

# %%
labels2 = OMEZarrImage(
    data=binary_blobs(length=64, volume_fraction=0.1, n_dim=3).astype('int8'),
    axes="zyx",
)
labels_multiscales2 = OMEZarrLabels(image=labels2)

# %% [markdown]
# ...and write the whole lot to disk.
# Again, when `overwrite=False`, the existing image data (and the already written label image `label_image_1`) will not be overwritten, but the new label image `label_image_2` will be added to the `labels` group as a new subgroup. 

# %%
parent_image.labels["label_image_2"] = labels_multiscales2
parent_image.to_ome_zarr("image_with_labels.zarr", overwrite=False)


