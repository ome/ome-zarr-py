import copy

import dask
import dask.array as da
import numpy as np
import pytest
import zarr
from dask.callbacks import Callback

from ome_zarr import USE_DASK_ARRAY_KWARGS
from ome_zarr.format import FormatV04, FormatV05
from ome_zarr.writer import write_image


@pytest.fixture(params=[None, 8192])
def shard_config(request):
    with zarr.config.set({"array.target_shard_size_bytes": request.param}):
        yield


@pytest.mark.skipif(not USE_DASK_ARRAY_KWARGS, reason="Requires Dask array kwargs")
@pytest.mark.parametrize("compute", [False, True])
@pytest.mark.parametrize("chunks", [(1, 8, 8), 4, None])
@pytest.mark.parametrize("per_level", [False, True])
def test_auto_sharding_roundtrip(tmp_path, compute, chunks, per_level, shard_config):
    data = np.arange(5 * 34 * 38, dtype=np.uint16).reshape(5, 34, 38)
    image = da.from_array(data, chunks=(2, 7, 9))
    options = {
        "shards": "auto",
        "config": {"order": "C"},
    }
    if chunks is not None:
        options["chunks"] = chunks
    if per_level:
        options = [
            options,
            {**options, "chunks": (1, 4, 4), "config": {"order": "F"}},
        ]
    original_options = copy.deepcopy(options)
    kwargs = dict(
        axes="zyx",
        fmt=FormatV05(),
        scale_factors=[2],
        method="nearest",
        scale={"z": 2, "y": 0.5, "x": 0.5},
        axes_units=dict.fromkeys("zyx", "micrometer"),
        name="auto-sharding",
    )
    executed = []
    with Callback(posttask=lambda key, *args: executed.append(key)):
        jobs = write_image(
            image,
            str(tmp_path / "auto.zarr"),
            storage_options=options,
            compute=compute,
            **kwargs,
        )
    if not compute:
        assert not executed
        assert len(jobs) == 2
        dask.compute(*jobs, scheduler="threads", num_workers=4)
    else:
        assert executed
        assert jobs == []
    assert options == original_options

    write_image(image, str(tmp_path / "reference.zarr"), **kwargs)
    result = zarr.open_group(tmp_path / "auto.zarr", mode="r")
    reference = zarr.open_group(tmp_path / "reference.zarr", mode="r")
    assert result.attrs.asdict() == reference.attrs.asdict()
    np.testing.assert_array_equal(result["s0"][:], data)
    for index in range(2):
        array = result[f"s{index}"]
        expected = reference[f"s{index}"]
        assert array.dtype == expected.dtype
        assert array.shape == expected.shape
        np.testing.assert_array_equal(array[:], expected[:])
        level_options = options[index] if per_level else options
        direct = zarr.create_array(
            zarr.storage.MemoryStore(),
            shape=array.shape,
            dtype=array.dtype,
            chunks=array.chunks,
            shards="auto",
            config=level_options["config"],
        )
        assert array.shards == direct.shards
        if not compute:
            # Every interior write boundary must be a shard boundary.
            for axis_chunks, shard in zip(jobs[index].chunks, array.shards):
                assert all(size == shard for size in axis_chunks[:-1])


def test_auto_sharding_legacy_dask(tmp_path, monkeypatch):
    monkeypatch.setattr("ome_zarr.writer.USE_DASK_ARRAY_KWARGS", False)
    with pytest.raises(
        ValueError, match=r"Automatic sharding requires Dask >= 2026\.3\.0"
    ):
        write_image(
            np.ones((8, 8), dtype=np.uint8),
            str(tmp_path / "legacy.zarr"),
            axes="yx",
            fmt=FormatV05(),
            scale_factors=[],
            storage_options={"shards": "auto"},
        )


@pytest.mark.skipif(not USE_DASK_ARRAY_KWARGS, reason="Requires Dask array kwargs")
def test_auto_sharding_issue640(tmp_path):
    data = np.random.default_rng(0).integers(0, 256, (100, 100, 100), dtype=np.uint8)
    options = {
        "chunks": (1, 100, 100),
        "shards": "auto",
        "config": {"array.target_shard_size_bytes": 1000},
    }
    # Preserve the issue's options, including its per-array config. Zarr decides
    # which settings it recognizes; OME-Zarr must not reinterpret them.
    direct = zarr.create_array(
        zarr.storage.MemoryStore(), shape=data.shape, dtype=data.dtype, **options
    )
    write_image(
        data, str(tmp_path / "issue640.zarr"), axes="zyx", storage_options=options
    )
    result = zarr.open_group(tmp_path / "issue640.zarr", mode="r")
    assert result["s0"].shards == direct.shards
    np.testing.assert_array_equal(result["s0"][:], data)


@pytest.mark.skipif(not USE_DASK_ARRAY_KWARGS, reason="Requires Dask array kwargs")
def test_auto_sharding_rejects_zarr_v2(tmp_path):
    with pytest.raises(ValueError, match="Zarr format 2"):
        write_image(
            np.ones((8, 8), dtype=np.uint8),
            str(tmp_path / "v2.zarr"),
            axes="yx",
            fmt=FormatV04(),
            scale_factors=[],
            storage_options={"shards": "auto"},
        )
