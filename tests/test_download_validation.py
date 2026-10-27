"""Download validation must remain active under optimized Python."""

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("optimized", [False, True], ids=["regular", "optimized"])
@pytest.mark.parametrize("existing_output", [False, True], ids=["invalid", "existing"])
def test_download_validation(optimized, existing_output, tmp_path):
    script = textwrap.dedent("""
        import sys
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        from ome_zarr.utils import download

        output = Path(sys.argv[1])
        existing = sys.argv[2] == "True"
        location = Mock(root_attrs={}) if existing else None
        node = Mock(zarr=SimpleNamespace(parts=lambda: ["image.zarr"]))
        reader = Mock(return_value=Mock(return_value=iter([node])))
        expected = (
            f"{output / 'image.zarr'} already exists!"
            if existing else "not a zarr: None"
        )
        expected_type = FileExistsError if existing else ValueError
        with (
            patch("ome_zarr.utils.parse_url", return_value=location),
            patch("ome_zarr.utils.Reader", reader),
            patch("ome_zarr.utils.Path.mkdir") as mkdir,
            patch("ome_zarr.utils.zarr.open_group") as open_group,
            patch("ome_zarr.utils.da.to_zarr") as to_zarr,
        ):
            try:
                download("image.zarr", str(output))
            except expected_type as error:
                if str(error) != expected:
                    raise RuntimeError(f"Unexpected message: {error}") from error
            else:
                raise RuntimeError("Download validation did not reject the input")
            mkdir.assert_not_called()
            open_group.assert_not_called()
            to_zarr.assert_not_called()
            node.write_metadata.assert_not_called()
            if not existing:
                reader.assert_not_called()
        """)
    if existing_output:
        target = tmp_path / "image.zarr"
        target.mkdir()
        sentinel = target / "sentinel.txt"
        sentinel.write_text("preserved")
    command = [sys.executable]
    if optimized:
        command.append("-O")
    subprocess.run(  # noqa: S603 - fixed script and current Python interpreter
        command + ["-c", script, str(tmp_path), str(existing_output)],
        check=True,
        capture_output=True,
        text=True,
    )
    if existing_output:
        assert sentinel.read_text() == "preserved"
        assert list(target.iterdir()) == [sentinel]
    else:
        assert list(tmp_path.iterdir()) == []
