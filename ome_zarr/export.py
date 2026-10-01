import glob
import logging
import os
import re
from enum import StrEnum
from pathlib import Path
from typing import Any

import dask.array as da
from dask.delayed import Delayed, delayed

from ome_zarr import OMEZarrImage
from ome_zarr.classes.image import OMEZarrMultiscaleBase

LOGGER = logging.getLogger(__name__)


class ExportFormat(StrEnum):
    TIFF_STACK = "tiff-stack"  # TIFF stack, uses ImageJ-style metadata
    OME_TIFF_STACK = "ome-tiff-stack"  # TIFF stack, uses OME-TIFF/OME-XML metadata
    # Could also add: TIFF, OME-TIFF, ...


# Axis order of the exported planes; OME-Zarr (>=0.4) mandates t, c, z, y, x
# See: https://ngff.openmicroscopy.org/0.5/#multiscale-md
TIFF_AXES_ORDER = "tczyx"

# Dtypes that can be stored with ImageJ metadata
IMAGEJ_DTYPES = {"uint8", "uint16", "int16", "float32"}

# Dtypes that OME-XML can describe (it has no 64-bit integer pixel types)
# See: https://ome-model.readthedocs.io/en/stable/ome-xml/
OME_DTYPES = {
    "bool",
    "int8",
    "int16",
    "int32",
    "uint8",
    "uint16",
    "uint32",
    "float32",
    "float64",
    "complex64",
    "complex128",
}

# OME-Zarr length units (UDUNITS-2 names) expressed in micrometers
# See: https://ngff.openmicroscopy.org/0.5/#axes-md
# Note: OME-XML uses micrometers as default
LENGTH_UNITS_IN_UM = {
    "angstrom": 1e-4,
    "picometer": 1e-6,
    "nanometer": 1e-3,
    "micrometer": 1.0,
    "micron": 1.0,
    "millimeter": 1e3,
    "centimeter": 1e4,
    "meter": 1e6,
}

# OME-Zarr time units (UDUNITS-2 names) expressed in seconds
# See: https://ngff.openmicroscopy.org/0.5/#axes-md
TIME_UNITS_IN_S = {
    "nanosecond": 1e-9,
    "microsecond": 1e-6,
    "millisecond": 1e-3,
    "second": 1.0,
    "minute": 60.0,
    "hour": 3600.0,
    "day": 86400.0,
}

# OME-Zarr units (UDUNITS-2 names) -> OME-XML unit symbols
# See: https://ome-model.readthedocs.io/en/2025.12.09/developers/ome-units.html
OME_UNITS = {
    "angstrom": "Å",
    "picometer": "pm",
    "nanometer": "nm",
    "micrometer": "µm",
    "micron": "µm",
    "millimeter": "mm",
    "centimeter": "cm",
    "meter": "m",
    "nanosecond": "ns",
    "microsecond": "µs",
    "millisecond": "ms",
    "second": "s",
    "minute": "min",
    "hour": "h",
    "day": "d",
}


def _tiff_metadata(image: OMEZarrImage, *, ome: bool, name: str) -> dict[str, Any]:
    """
    Build the ``tifffile.imwrite`` keyword arguments carrying the voxel size.

    Both variants write the baseline XResolution/YResolution tags in pixels per
    cm (ResolutionUnit CENTIMETER) when x and y have known length units, so
    generic TIFF readers (e.g. exiftool) see the pixel size. Without known
    units, the tags hold pixels per scale unit (ResolutionUnit NONE).

    With ``ome=False``, an ImageJ description is written the way both ImageJ/Fiji
    and Bio-Formats interpret it: lengths in micrometers (``unit=um``, z
    ``spacing``) and the frame interval in seconds.

    With ``ome=True``, an OME-XML description holds the image ``name``, and the
    physical sizes and time increment in their original units. Axes without a
    known unit are left out, since OME would otherwise assume micrometers.

    Raises a ValueError for dtypes the chosen metadata cannot store.
    """
    dtype = str(image.data.dtype)
    if ome and dtype not in OME_DTYPES:
        raise ValueError(
            f"OME-TIFF cannot store dtype {dtype} (supported: {sorted(OME_DTYPES)})"
        )
    if not ome and dtype not in IMAGEJ_DTYPES:
        raise ValueError(
            f"ImageJ TIFF cannot store dtype {dtype} "
            f"(supported: {sorted(IMAGEJ_DTYPES)}); "
            f"export as '{ExportFormat.OME_TIFF_STACK}' instead"
        )

    scale = image.scale or {}
    units = image.axes_units or {}
    for ax in scale.keys() & set("tzyx"):
        if ax in units and units[ax] not in OME_UNITS:
            LOGGER.warning(
                "Unsupported unit %r for axis %r; it is not written to the TIFF",
                units[ax],
                ax,
            )

    def in_um(ax: str) -> float | None:
        unit = units.get(ax)
        if ax in scale and unit in LENGTH_UNITS_IN_UM:
            return scale[ax] * LENGTH_UNITS_IN_UM[unit]
        return None

    x_um, y_um, z_um = in_um("x"), in_um("y"), in_um("z")
    if x_um is not None and y_um is not None:
        calibrated = True
        kwargs: dict[str, Any] = {
            "resolution": (1e4 / x_um, 1e4 / y_um),
            "resolutionunit": "CENTIMETER",
        }
    else:
        calibrated = False
        kwargs = {
            "resolution": (1 / scale.get("x", 1.0), 1 / scale.get("y", 1.0)),
            "resolutionunit": "NONE",
        }

    if ome:
        # Each plane is written as a standalone single-plane OME-TIFF, so readers
        # such as Bio-Formats open every file as a separate image, not the folder
        # as one dataset. A multi-file OME-TIFF would need OME-XML in each file
        # that references all files by UUID (TiffData/UUID/FileName), which
        # tifffile does not generate. See "Multi-file OME-TIFF" in
        # https://docs.openmicroscopy.org/ome-model/5.6.3/ome-tiff/specification.html
        metadata: dict[str, Any] = {"axes": "YX", "Name": name}
        for ax, key in [
            ("x", "PhysicalSizeX"),
            ("y", "PhysicalSizeY"),
            ("z", "PhysicalSizeZ"),
            ("t", "TimeIncrement"),
        ]:
            if ax in scale and units.get(ax) in OME_UNITS:
                metadata[key] = scale[ax]
                metadata[f"{key}Unit"] = OME_UNITS[units[ax]]
        return {**kwargs, "ome": True, "metadata": metadata}

    # not OME, so use ImageJ metadata style
    # ImageJ scales px/cm resolution tags to micrometers for unit=um,
    # and Bio-Formats ALWAYS reads `spacing` in micrometers and
    # `finterval` in seconds
    imagej: dict[str, Any] = {"axes": "YX"}
    if calibrated:
        imagej["unit"] = "um"
    if "z" in scale:
        imagej["spacing"] = z_um if calibrated and z_um is not None else scale["z"]
    if "t" in scale:
        if units.get("t") in TIME_UNITS_IN_S:
            imagej["finterval"] = scale["t"] * TIME_UNITS_IN_S[units["t"]]
            imagej["tunit"] = "sec"
        else:
            imagej["finterval"] = scale["t"]
    return {**kwargs, "imagej": True, "metadata": imagej}


def _write_tiff_stack(
    image: OMEZarrImage,
    output_dir: Path,
    *,
    overwrite: bool,
    ome: bool,
) -> list[Delayed]:
    """
    Build the dask tasks writing a single-resolution image as a folder of 2D
    TIFF files, one task per plane.

    With ``ome=True``, each file carries OME-XML metadata; otherwise ImageJ metadata.

    Files are named ``<name>_t<i>_c<i>_z<i>.tif`` (``.ome.tif`` for ``ome=True``),
    with one index per non-YX axis present in the image.
    Each index is zero-padded to the number of digits of that axis' largest index
    (e.g. ``z000`` to ``z511`` for 512 planes), so that files sort in t, c, z order.

    With ``overwrite``, existing planes of a stack with the same name are
    removed first; other files in ``output_dir`` are left untouched.
    """
    import numpy as np
    import tifffile

    axes = list(image.axes)
    if unknown := set(axes) - set(TIFF_AXES_ORDER):
        raise ValueError(
            f"Cannot export axes {unknown} to TIFF. "
            f"Supported axes are: {list(TIFF_AXES_ORDER)}"
        )
    if not {"y", "x"} <= set(axes):
        raise ValueError(f"TIFF export requires 'y' and 'x' axes, got {axes}")

    # reorder to canonical TCZYX order (already the case for spec-compliant OME-Zarr)
    ordered_axes = [ax for ax in TIFF_AXES_ORDER if ax in axes]
    data = da.transpose(image.data, [axes.index(ax) for ax in ordered_axes])

    # one chunk per YX plane, so every dask block maps to exactly one file
    plane_axes = ordered_axes[:-2]
    data = data.rechunk((1,) * len(plane_axes) + data.shape[-2:])

    # multiscale names may be zarr paths (e.g. "/"), so make them filename-safe
    stem = re.sub(r'[\\/:*?"<>|]+', "_", image.name).strip("_. ") or "image"
    tiff_kwargs = _tiff_metadata(image, ome=ome, name=stem)
    # calculate axis widths for zero-padding the plane indices in the filenames
    widths = [len(str(n - 1)) for _, n in zip(plane_axes, data.shape)]

    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"TIFF stack output must be a directory: {output_dir}")
    # the OME-TIFF specification requires the .ome.tif(f) extension
    ext = ".ome.tif" if ome else ".tif"
    pattern = glob.escape(stem)
    existing = [
        path
        for path in [
            *output_dir.glob(f"{pattern}{ext}"),
            *output_dir.glob(f"{pattern}_*{ext}"),
        ]
        # leave an OME-TIFF stack of the same name alone when writing plain TIFFs
        if ome or not path.name.endswith(".ome.tif")
    ]
    if existing:
        if not overwrite:
            raise FileExistsError(f"TIFF stack '{stem}' already exists in {output_dir}")
        for path in existing:
            path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    def write_plane(plane: np.ndarray, path: Path) -> None:
        tifffile.imwrite(
            path,
            plane.reshape(plane.shape[-2:]),
            photometric="minisblack",
            **tiff_kwargs,
        )

    blocks = data.to_delayed()
    writes = []
    for index in np.ndindex(blocks.shape[:-2]):
        suffix = "".join(
            f"_{ax}{i:0{w}d}" for ax, i, w in zip(plane_axes, index, widths)
        )
        path = output_dir / f"{stem}{suffix}{ext}"
        writes.append(delayed(write_plane)(blocks[index + (0, 0)], path))
    return writes


def export(
    image: OMEZarrMultiscaleBase,
    export_location: str | os.PathLike,
    export_format: ExportFormat | str,
    *,
    overwrite: bool = False,
    level: int = 0,
) -> list[Delayed]:
    """
    Build the dask tasks exporting a multiscale image to another file format.

    The output location is prepared immediately (including removing an
    existing export with ``overwrite``), but the data is only written when the
    returned tasks are computed, e.g. with ``dask.compute(*export(...))``.
    This lets the caller choose the scheduler and show progress (e.g. with
    :class:`dask.diagnostics.ProgressBar`).

    Parameters
    ----------
    image : OMEZarrMultiscaleBase
        The multiscale image (or labels) to export, e.g. as loaded with
        :meth:`OMEZarrMultiscale.from_ome_zarr`.
    export_location : str or os.PathLike
        Where to write the export. For TIFF stacks this is a directory, which
        is created if needed; each YX plane is written to its own file,
        named after the image.
    export_format : ExportFormat or str
        The output format:

        - ``"tiff-stack"``: 2D TIFF files with ImageJ metadata, read by
          ImageJ/Fiji and Bio-Formats. Only supports the dtypes ImageJ can
          store (uint8, uint16, int16 and float32).
        - ``"ome-tiff-stack"``: 2D OME-TIFF files with OME-XML metadata,
          supporting all dtypes except 64-bit integers, which OME-XML cannot
          describe. Each file is a standalone OME-TIFF, not part of a
          multi-file OME-TIFF dataset.

        An unsupported dtype raises :class:`ValueError`.

        Both also write the baseline TIFF resolution tags.
    overwrite : bool
        Replace an existing export of the same image at ``export_location``.
        If False, an existing export raises :class:`FileExistsError`.
    level : int
        Pyramid level to export. 0 is the full-resolution level; negative
        values count from the lowest resolution, i.e. -1 is the smallest level.

    Returns
    -------
    list of dask.delayed.Delayed
        The write tasks; for TIFF stacks, one per plane.
    """
    output_path = Path(export_location)

    n_levels = len(image.images)
    if not -n_levels <= level < n_levels:
        raise IndexError(
            f"Pyramid level {level} out of range for image with {n_levels} levels"
        )

    if isinstance(export_format, str):
        export_format = ExportFormat(export_format)

    match export_format:
        case ExportFormat.TIFF_STACK | ExportFormat.OME_TIFF_STACK:
            return _write_tiff_stack(
                image.images[level],
                output_path,
                overwrite=overwrite,
                ome=export_format == ExportFormat.OME_TIFF_STACK,
            )
        case _:
            raise ValueError(f"Unsupported output format: {export_format}")
