# %% [markdown]
# # Read OME-ZARR images
# (basic:read)=
# 
# This sample code reads an image stored on remote s3 server,
# but the same code can be used to read data on a local file system.
# In either case, the data is exposed as an instance of the {py:class}`ome_zarr.classes.image.OMEZarrMultiscale` class, which provides access to the multiscale levels and metadata of the OME-ZARR image.

# %%
from ome_zarr import OMEZarrMultiscale

url = "https://livingobjects.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr"

ngff_image = OMEZarrMultiscale.from_ome_zarr(url)

# %% [markdown]
# You can access the multiscale levels by inspecting the `images` attributes of the {py:class}`ome_zarr.classes.image.OMEZarrMultiscale` object,
# which is a list of {py:class}`ome_zarr.classes.image.OMEZarrImage` objects.

# %%
ngff_image.images

# %% [markdown]
# And of course, retrieve the data as a `dask` array using the `data` attribute of each image:

# %%
ngff_image.images[0].data

# %% [markdown]
# You can check whether label images were attached to this image by inspecting the `labels` attribute of the {py:class}`ome_zarr.classes.image.OMEZarrMultiscale` object,
# which is a dictionary mapping label image names to {py:class}`ome_zarr.classes.image.OMEZarrLabels` objects.

# %%
ngff_image.labels

# %% [markdown]
# ## Direct read
# 
# The code below here demonstrates an alternative, equally functional API for reading OME-ZARR images.
# 
# This sample code reads an image stored on remote s3 server,
# but the same code can be used to read data on a local file system.
# In either case, the data is exposed as [`dask` arrays](https://docs.dask.org/en/stable/array.html);
# 
# You can obtain a list of "nodes" which include all arrays stored in the group:

# %%
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

# read the image data
reader = Reader(parse_url(url))
# nodes may include images, labels etc
nodes = list(reader())
nodes

# %% [markdown]
# The first node will be the image pixel data;
# Since this group is again an ome-zarr multiscales object, it consists of several arrays that represent the different resolution levels:

# %%
image_node = nodes[0]

multiscales = image_node.data
multiscales

# %% [markdown]
# The first entry in this list represents the 0-th resolution level, and is the highest resolution data.

# %%
multiscales[0]

# %%



