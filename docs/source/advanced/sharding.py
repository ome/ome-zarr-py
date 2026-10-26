# %% [markdown]
# # Sharding
# (advanced:sharding)=
#
# [Sharding](https://zarr.dev/zeps/accepted/ZEP0002.html) is a technique in [Zarr](https://zarr.dev/) for bundling multiple chunks into a single storage unit.
# In terms of storage, this can drastically reduce the number of files written,
# and thus reduce the overhead of file system operations and streaming data from remote storages.
#
# ![Sharding](https://zarr.dev/zeps/assets/images/sharding.png)
#
# In the following tutorial, we will walk through how to use sharding with `ome-zarr-py` and Dask arrays.
#

# %%
import os
import shutil

import dask.array as da

from ome_zarr.writer import write_image

# %% [markdown]
# First, we create some random data as a Dask array:

# %%
array = da.random.random((1, 3, 128, 128, 128), chunks=(1, 1, 16, 16, 16))

# %% [markdown]
# We now write the data to a local Zarr store with sharding enabled;
# This can be done by adding the `shards` key to the `storage_options` argument of {py:func}`ome_zarr.writer.write_image`.
# The storage options generally accepts keywords as listed by the zarr documentation:
# - For dask > 2026.3.0, all keywords are documented under [https://zarr.readthedocs.io/en/stable/api/zarr/create/#zarr.create_array](https://zarr.readthedocs.io/en/stable/api/zarr/create/#zarr.create_array).
# - For dask <= 2025.11.0 sharding is **not** available; All other keywords are documented under [https://zarr.readthedocs.io/en/stable/api/zarr/create/#zarr.create](https://zarr.readthedocs.io/en/stable/api/zarr/create/#zarr.create)
#
# ```{note}
# The fact that sharding is only available for `dask > 2026.3.0`, conversely, means that *reading* sharded data is also not possible with `dask <= 2025.11.0`.
# ```
#
# In any case, the size of a shard has to be an *integer multiple* of the chunk size.
# In this example, we set the shard size to be (1, 1, 32, 32, 32), which means that each shard will contain 8 chunks of data:

# %%
if os.path.exists("test.ome.zarr"):
    shutil.rmtree("test.ome.zarr")

write_image(
    array,
    group="test.ome.zarr",
    storage_options={"shards": [1, 1, 32, 32, 32]},
)

# %% [markdown]
# If you want or need, you can also set the shard size explicitly for every resolution level.
# I.e., in the following case, these chunk sizes are realized for the different resolution level:
# - Level 0: Shard size is ``(1, 1, 128, 128, 128)``, which means that each shard will contain 64 chunks of data.
# - Level 1: Shard size is ``(1, 1, 64, 64, 64)``, which means that each shard will contain 8 chunks of data.
# - Level 2: Shard size is ``(1, 1, 32, 32, 32)``, which means that each shard will contain 1 chunk of data.

# %%
if os.path.exists("test.ome.zarr"):
    shutil.rmtree("test.ome.zarr")

write_image(
    array,
    group="test.ome.zarr",
    scale_factors=[
        {"t": 1, "c": 1, "z": 2, "y": 2, "x": 2},
        {"t": 1, "c": 1, "z": 4, "y": 4, "x": 4},
    ],
    storage_options=[
        {"shards": [1, 1, 128, 128, 128]},
        {"shards": [1, 1, 64, 64, 64]},
        {"shards": [1, 1, 32, 32, 32]},
    ],
)

# %% [markdown]
# ## Difference in numbers
#
# Let's check how many files are present in the output directory *with sharding enabled*:

# %%
if os.path.exists("test.ome.zarr"):
    shutil.rmtree("test.ome.zarr")

write_image(
    array,
    group="test.ome.zarr",
    storage_options={"shards": [1, 1, 32, 32, 32]},
)

n_files = sum(len(files) for _, _, files in os.walk("test.ome.zarr"))
print(f"Number of files in the output directory: {n_files}")

# %% [markdown]
# Now we repeat the same process but with sharding disabled (by removing the `shards` key from the `storage_options`):

# %%
if os.path.exists("test.ome.zarr"):
    shutil.rmtree("test.ome.zarr")

write_image(
    array,
    group="test.ome.zarr",
    # storage_options={"shards": [1, 1, 64, 64, 64]},
)

n_files = sum(len(files) for _, _, files in os.walk("test.ome.zarr"))
print(f"Number of files in the output directory: {n_files}")

# %% [markdown]
# We can see a *very* notable effect in the amount of files written.
#
# ## What shard sizes are allowed?
#
# Besides the requirement that the shard size has to be an integer multiple of the chunk size,
# you are free to set the shard size to any value that you think is appropriate for your use case.
#
# For instance, if your image is `(1, 1, 128, 128, 128)` and your chunk size is `(1, 1, 32, 32, 32)`,
# any of the following shard sizes would be valid:
# - `(1, 1, 32, 32, 32)` (one chunk per shard)
# - `(1, 1, 64, 32, 96)` (mixed multiples of the chunk size)
# - `(1, 1, 96, 96, 96)` (multiple chunks per shard)
# - `(1, 1, 128, 128, 128)` (one shard for the whole image)

# %%
if os.path.exists("test.ome.zarr"):
    shutil.rmtree("test.ome.zarr")

write_image(
    array,
    group="test.ome.zarr",
    storage_options={"shards": [1, 1, 128, 128, 128]},
)

n_files = sum(len(files) for _, _, files in os.walk("test.ome.zarr"))
print(f"Number of files in the output directory: {n_files}")
