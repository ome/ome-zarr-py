# %% [markdown]
# # How to handle delayed writing
#
# Writing large images can take a while, because ome-zarr-py computes the
# multiscale pyramid and writes every chunk to storage.
# Under the hood, this is done with [dask](https://docs.dask.org).
#
# By default (`compute=True`),
# {py:func}`ome_zarr.classes.image.OMEZarrMultiscale.to_ome_zarr` does this immediately and returns an empty list.
# With `compute=False`, it only writes the metadata and returns the remaining work as a list of dask objects, which you can run later yourself.

# %%
import dask
import numpy as np
from dask.diagnostics import ProgressBar

from ome_zarr import OMEZarrImage, OMEZarrMultiscale

# Create a random image to write.
# To test with your own image, pass any numpy or dask array as `data`,
# for example `data = tifffile.imread("image.tif")` and change `axes` to match `data.shape`.
# To see the progress bar move, use an image of a few hundred MB,
# but smaller than your computer's memory, because `tifffile.imread` loads the whole file into memory.
rng = np.random.default_rng(0)
data = rng.poisson(lam=10, size=(64, 256, 256)).astype(np.uint8)

# %% [markdown]
# ## A progress bar for a single image
#
# dask's {py:class}`~dask.diagnostics.ProgressBar` shows the progress of any dask
# computation that runs inside it. This includes `to_ome_zarr` with the default `compute=True`:

# %%
image = OMEZarrImage(data, axes="zyx", name="image_a")
multiscale = OMEZarrMultiscale(image, scale_factors=[2, 4])

with ProgressBar():
    multiscale.to_ome_zarr("progress_image_a.ome.zarr")

# %% [markdown]
# With `compute=True`, every call to `to_ome_zarr` starts its own computation.
# Writing several images will show one progress bar per image.
# For more control, use `compute=False`.
#
# ## Writing with `compute=False`
#
# Instead of writing the data, `to_ome_zarr` returns one dask object per
# resolution level:

# %%
image = OMEZarrImage(data, axes="zyx", name="image_a")
multiscale = OMEZarrMultiscale(image, scale_factors=[2, 4])

jobs = multiscale.to_ome_zarr("delayed_image_a.ome.zarr", compute=False)
len(jobs)

# %% [markdown]
# At this point, only the metadata has been written.
# The image data is written when the jobs are computed.
#
# ## Running the jobs with a progress bar
#
# Run the jobs yourself with `dask.compute`, inside a
# {py:class}`~dask.diagnostics.ProgressBar`:

# %%
with ProgressBar():
    dask.compute(*jobs)

# %% [markdown]
# ## Writing several images at once
#
# The jobs of several images can be collected and computed together.
# This way, a single progress bar shows the progress of a batch.

# %%
jobs = []
for name in ["image_b", "image_c"]:
    data = rng.poisson(lam=10, size=(64, 256, 256)).astype(np.uint8)
    image = OMEZarrImage(data, axes="zyx", name=name)
    multiscale = OMEZarrMultiscale(image, scale_factors=[2, 4])
    jobs += multiscale.to_ome_zarr(f"delayed_{name}.ome.zarr", compute=False)

with ProgressBar():
    dask.compute(*jobs)

# %% [markdown]
# The same applies to scenes: {py:meth}`ome_zarr.classes.scene.OMEZarrScene.to_ome_zarr`
# also accepts `compute=False` and returns the jobs of all its images
# (see [Create Scenes](advanced:create-scenes)).
#
# ```{note}
# {py:class}`~dask.diagnostics.ProgressBar` works with dask's local schedulers
# (the default). If you use a [dask.distributed](https://distributed.dask.org)
# cluster, you should use {py:func}`dask.distributed.progress` instead.
# ```
