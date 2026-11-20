import dask
import numpy as np
import pytest
import tifffile
import zarr

from ome_zarr import OMEZarrMultiscale
from ome_zarr.export import export
from ome_zarr.writer import write_image


def make_image(path, dtype="uint16", space_unit="micrometer", time_unit="minute"):
    """Write a small tzyx OME-Zarr image with scale [1.5, 2, 0.5, 0.25]."""
    axes = [{"name": "t", "type": "time"}] + [
        {"name": ax, "type": "space"} for ax in "zyx"
    ]
    if time_unit:
        axes[0]["unit"] = time_unit
    if space_unit:
        for ax in axes[1:]:
            ax["unit"] = space_unit
    group = zarr.open_group(str(path), mode="w")
    data = np.zeros((2, 3, 8, 8), dtype=dtype)
    write_image(data, group, axes=axes, scale_factors=[])
    ome = group.attrs["ome"]
    multiscale = ome["multiscales"][0]
    multiscale["axes"] = axes
    multiscale["datasets"][0]["coordinateTransformations"][0]["scale"] = [
        1.5,
        2.0,
        0.5,
        0.25,
    ]
    group.attrs["ome"] = ome
    return OMEZarrMultiscale.from_ome_zarr(str(path))


def run_export(*args, **kwargs):
    dask.compute(*export(*args, **kwargs))


def read_first_plane(output_dir):
    files = sorted(output_dir.glob("*.tif"))
    assert len(files) == 6
    return tifffile.TiffFile(files[0])


@pytest.mark.parametrize(
    "space_unit, factor", [("micrometer", 1.0), ("nanometer", 1e-3), ("meter", 1e6)]
)
def test_imagej_calibration(tmp_path, space_unit, factor):
    image = make_image(tmp_path / "in.ome.zarr", space_unit=space_unit)
    run_export(image, tmp_path / "out", "tiff-stack")

    with read_first_plane(tmp_path / "out") as tif:
        page = tif.pages[0]
        assert page.tags["ResolutionUnit"].value == tifffile.RESUNIT.CENTIMETER
        # pixels per cm
        assert page.get_resolution() == pytest.approx(
            (1e4 / (0.25 * factor), 1e4 / (0.5 * factor))
        )
        # ImageJ and Bio-Formats read spacing in um and finterval in seconds
        meta = tif.imagej_metadata
        assert meta["unit"] == "um"
        assert meta["spacing"] == pytest.approx(2.0 * factor)
        assert meta["finterval"] == pytest.approx(90.0)
        assert meta["tunit"] == "sec"


def test_ome_calibration(tmp_path):
    image = make_image(tmp_path / "in.ome.zarr", space_unit="nanometer")
    run_export(image, tmp_path / "out", "ome-tiff-stack")

    with read_first_plane(tmp_path / "out") as tif:
        assert tif.is_ome
        page = tif.pages[0]
        assert page.tags["ResolutionUnit"].value == tifffile.RESUNIT.CENTIMETER
        ome_image = tifffile.xml2dict(tif.ome_metadata)["OME"]["Image"]
        assert ome_image["Name"] == "image"
        pixels = ome_image["Pixels"]
        assert pixels["PhysicalSizeX"] == 0.25
        assert pixels["PhysicalSizeXUnit"] == "nm"
        assert pixels["PhysicalSizeY"] == 0.5
        assert pixels["PhysicalSizeZ"] == 2.0
        assert pixels["PhysicalSizeZUnit"] == "nm"
        assert pixels["TimeIncrement"] == 1.5
        assert pixels["TimeIncrementUnit"] == "min"


def test_filenames_padded_to_axis_length(tmp_path):
    image = make_image(tmp_path / "in.ome.zarr")
    image.images[0].data = image.images[0].data.repeat(4, axis=1)  # 12 z-planes
    run_export(image, tmp_path / "out", "tiff-stack")

    names = sorted(p.name for p in (tmp_path / "out").glob("*.tif"))
    assert len(names) == 24
    assert names[:3] == ["image_t0_z00.tif", "image_t0_z01.tif", "image_t0_z02.tif"]
    assert names[-1] == "image_t1_z11.tif"


def test_export_is_lazy(tmp_path):
    image = make_image(tmp_path / "in.ome.zarr")
    out = tmp_path / "out"
    tasks = export(image, out, "tiff-stack")
    assert len(tasks) == 6
    assert list(out.glob("*.tif")) == []

    dask.compute(*tasks)
    assert len(list(out.glob("*.tif"))) == 6


def test_ome_tiff_extension(tmp_path):
    image = make_image(tmp_path / "in.ome.zarr")
    out = tmp_path / "out"
    run_export(image, out, "ome-tiff-stack")
    run_export(image, out, "tiff-stack")

    ome_files = sorted(out.glob("*.ome.tif"))
    assert len(ome_files) == 6
    assert ome_files[0].name == "image_t0_z0.ome.tif"
    assert len(list(out.glob("*.tif"))) == 12

    # overwriting one stack leaves the other alone
    run_export(image, out, "tiff-stack", overwrite=True)
    assert len(list(out.glob("*.ome.tif"))) == 6
    with pytest.raises(FileExistsError):
        export(image, out, "ome-tiff-stack")


def test_uncalibrated(tmp_path):
    image = make_image(tmp_path / "in.ome.zarr", space_unit=None, time_unit=None)
    run_export(image, tmp_path / "imagej", "tiff-stack")
    run_export(image, tmp_path / "ome", "ome-tiff-stack")

    with read_first_plane(tmp_path / "imagej") as tif:
        page = tif.pages[0]
        assert page.tags["ResolutionUnit"].value == tifffile.RESUNIT.NONE
        assert page.get_resolution() == pytest.approx((4.0, 2.0))
        assert "unit" not in tif.imagej_metadata
        assert tif.imagej_metadata["spacing"] == 2.0

    # OME assumes micrometers for sizes without a unit, so none are written
    with read_first_plane(tmp_path / "ome") as tif:
        pixels = tifffile.xml2dict(tif.ome_metadata)["OME"]["Image"]["Pixels"]
        assert "PhysicalSizeX" not in pixels
        assert "TimeIncrement" not in pixels


@pytest.mark.parametrize(
    "export_format, dtype, supported_dtype",
    [("tiff-stack", "float64", "uint16"), ("ome-tiff-stack", "int64", "float64")],
)
def test_unsupported_dtype(tmp_path, export_format, dtype, supported_dtype):
    out = tmp_path / "out"
    image = make_image(tmp_path / "supported.ome.zarr", dtype=supported_dtype)
    run_export(image, out, export_format)

    # raised before an existing stack is removed, even with overwrite
    image = make_image(tmp_path / "unsupported.ome.zarr", dtype=dtype)
    with pytest.raises(ValueError, match=dtype):
        export(image, out, export_format, overwrite=True)
    with read_first_plane(out) as tif:
        assert tif.pages[0].dtype == supported_dtype
