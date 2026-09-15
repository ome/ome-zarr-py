# %% [markdown]
# # Create Scenes
#
# (advanced:create-scenes)=
#
# This tutorial demonstrates basic usage around writing and reading [Ngff Scenes](https://ngff.openmicroscopy.org/specifications/dev/index.html#scene-metadata) using the `OMEZarrScene` class.

# %%
from ome_zarr_models.v06.coordinate_transforms import CoordinateSystem, Translation
from skimage import data

from ome_zarr import OMEZarrImage, OMEZarrMultiscale, OMEZarrScene

# %% [markdown]
# We create some sample data, which we will store as a tiled layout in a scene zarr group:

# %%
example_image = data.human_mitosis()
example_image.shape

# %% [markdown]
# First, we cut the image into four tiles and convert them into instances of `OMEZarrMultiscale`:

# %%
img1 = OMEZarrImage(data=example_image[:256, :256], axes=["y", "x"], name="img1")
img2 = OMEZarrImage(data=example_image[256:, :256], axes=["y", "x"], name="img2")
img3 = OMEZarrImage(data=example_image[:256, 256:], axes=["y", "x"], name="img3")
img4 = OMEZarrImage(data=example_image[256:, 256:], axes=["y", "x"], name="img4")

img1_ms = OMEZarrMultiscale(img1)
img2_ms = OMEZarrMultiscale(img2)
img3_ms = OMEZarrMultiscale(img3)
img4_ms = OMEZarrMultiscale(img4)

# %% [markdown]
# Next, we need to define a coordinate system into which all images are projected.
# This is defined in accordance with the NGFF [coordinate systems specification](https://ngff.openmicroscopy.org/specifications/dev/index.html#coordinatesystems-metadata).
# The coordinate systems can be defined using the `CoordinateSystem` class:

# %%
coordinate_system = CoordinateSystem.model_validate({
    "name": "world",
    "axes": [
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ],
})

# %% [markdown]
# ::::{hint}
# `CoordinateSystem` is a [pydantic](https://pydantic.dev) class.
# This means that it can be instantiated either from a to-be validated dictionary or from keyword arguments and subfields:
# ```python
# coordinate_system = CoordinateSystem.model_validate({...})
# coordinate_system = CoordinateSystem(name="world", axes=[...])
# ```
#
# In the second case, the `axes` argument would need to be populated with the respective `Axis` instances.
#
# ::::
#
# We then define translations that move each tile into the appropriate position in the world coordinate system.
# In this example, these are simple translations in the y and x dimensions:

# %%
coordinate_transformations = [
    Translation.model_validate({
        "type": "translation",
        "translation": [0, 0],
        "input": {"path": "img1", "name": "physical"},
        "output": {"name": "world"},
    }),
    Translation.model_validate({
        "type": "translation",
        "translation": [256, 0],
        "input": {"path": "img2", "name": "physical"},
        "output": {"name": "world"},
    }),
    Translation.model_validate({
        "type": "translation",
        "translation": [0, 256],
        "input": {"path": "img3", "name": "physical"},
        "output": {"name": "world"},
    }),
    Translation.model_validate({
        "type": "translation",
        "translation": [256, 256],
        "input": {"path": "img4", "name": "physical"},
        "output": {"name": "world"},
    }),
]

# %% [markdown]
# We can then create and write a scene like this:

# %%
scene = OMEZarrScene(
    images=[img1_ms, img2_ms, img3_ms, img4_ms],
    coordinate_systems=[coordinate_system],
    coordinate_transformations=coordinate_transformations
)

scene.to_ome_zarr("test_example_scene.zarr", overwrite=True)

# %% [markdown]
# ....and load it back like this:

# %%
loaded_scene = OMEZarrScene.from_ome_zarr("test_example_scene.zarr")
loaded_scene.coordinate_transformations

# %%
