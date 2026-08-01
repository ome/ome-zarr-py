# %% [markdown]
# # Image-labels metadata
# 
# To ease the interpretation of label images and make them identifiable as label images,
# it is helpful to add the [`image-label` metadata](https://ngff.openmicroscopy.org/specifications/0.5/index.html#labels-metadata) to the written label image data.
# The {py:class}`ome_zarr.classes.image.OMEZarrLabels` provides an easy entrypoint to pass this metadata when writing label images to disk and to read it back in when loading an image with labels from disk.
# 
# Similar to the [previous notebook](advanced:labels:adding_labels), we will start by creating a parent image and a dummy label image, to which we add some metadata.

# %%
import numpy as np
from skimage.data import binary_blobs

from ome_zarr import OMEZarrImage, OMEZarrLabels, OMEZarrMultiscale

# %%
rng = np.random.default_rng(0)
data = rng.poisson(10, size=(64, 64, 64)).astype(np.uint8)

ngff_image = OMEZarrImage(data, axes="zyx", name="image")
ngff_multiscales = OMEZarrMultiscale(image=ngff_image)

# %%
labels = OMEZarrImage(
    data=binary_blobs(length=64, volume_fraction=0.1, n_dim=3).astype('int8'),
    axes="zyx",
)
labels_multiscales = OMEZarrLabels(image=labels)

# %% [markdown]
# ## Composing the image-label metadata
# 
# In Python context, the `image-label` metadata is a dictionary composed of the following fields:
# - `colors`: Consists of the actual integer label value and a list of rgba color values.
# - `properties`: Unspecified, per-object key-value pairs. The `label-value` property links the metadata to the actual label value in the label image.
# - `source`: The source of the label image. Automatically written by the writer tool.
# - `version`: The version of the metadata schema. Automatically written by the writer tool.
# 
# In our example (a binary label image), we need only provide the metadata for two kinds of objects (i.e., the background and the foreground):

# %%
colors = [
    {"label-value": 0, "rgba": [0, 0, 0, 255]},
    {"label-value": 1, "rgba": [255, 255, 255, 255]},
]

properties = [
    {"label-value": 0, "class": "background"},
    {"label-value": 1, "class": "foreground"},
]

# %% [markdown]
# We can now pass this metadata to the respective field of the {py:class}`ome_zarr.classes.image.OMEZarrLabels` object and write it to disk:

# %%
labels_multiscales.image_label = {
  "image-label": {
    "colors": colors,
    "properties": properties
  }
}

ngff_multiscales.labels = {"test_labels": labels_multiscales}

# write to disk
ngff_multiscales.to_ome_zarr("labels_metadata_example.zarr")

# %% [markdown]
# ## Reading the image-label metadata
# 
# When reading an ome-zarr with labels from disk, the `image-label` metadata is automatically parsed and made available as part of the {py:class}`ome_zarr.classes.image.OMEZarrMultiscale` object.
# We can retrieve the labels image from the `labels` attribute of the parent `OMEZarrMultiscale` image:

# %%
image = OMEZarrMultiscale.from_ome_zarr("labels_metadata_example.zarr")
image.labels["test_labels"].image_label

# %% [markdown]
# This can be converted into a more human-readable format using the `model_dump()` method of the `LabelBase` class:

# %%
image.labels["test_labels"].image_label.model_dump()

# %% [markdown]
# Alternatively, it is possible to read only the label image (using the {py:class}`ome_zarr.classes.image.OMEZarrLabels`) without ingesting the parent image and its metadata:

# %%
label_image = image = OMEZarrLabels.from_ome_zarr("labels_metadata_example.zarr/labels/test_labels")
label_image.image_label.model_dump()


