# %% [markdown]
# # Create a plate

# %%
from ome_zarr import OMEZarrHCSPlate, OMEZarrImage, OMEZarrMultiscale
import dask.array as da

# %%
random_data = da.random.random((1024, 1024))

# %%
img = OMEZarrImage(data=random_data, axes="yx")
ms = OMEZarrMultiscale(image=img)

# %%
plate = OMEZarrHCSPlate(
    images= {
        ("A", "1"): [ms]
    })

# %%
plate.to_ome_zarr("test_hcs.ome.zarr", overwrite=True)

# %%
OMEZarrHCSPlate.from_ome_zarr("test_hcs.ome.zarr")

# %%
plate_loaded = OMEZarrHCSPlate.from_ome_zarr("https://livingobjects.ebi.ac.uk/idr/zarr/v0.5/idr0010/76-45.ome.zarr")

# %%
