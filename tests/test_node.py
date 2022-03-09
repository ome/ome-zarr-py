import pytest
import zarr
from numpy import zeros

from ome_zarr.data import create_zarr
from ome_zarr.format import FormatV01, FormatV02, FormatV03
from ome_zarr.io import parse_url
from ome_zarr.reader import Label, Labels, Multiscales, Node, Plate, Well
from ome_zarr.writer import write_image, write_plate_metadata, write_well_metadata


class TestNode:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = tmpdir.mkdir("data")
        create_zarr(str(self.path))

    def test_image(self):
        node = Node(parse_url(str(self.path)), list())
        assert node.data
        assert node.metadata
        assert len(node.specs) == 2
        assert isinstance(node.specs[0], Multiscales)

    def test_labels(self):
        filename = str(self.path.join("labels"))
        node = Node(parse_url(filename), list())
        assert not node.data
        assert not node.metadata
        assert len(node.specs) == 1
        assert isinstance(node.specs[0], Labels)

    def test_label(self):
        filename = str(self.path.join("labels", "coins"))
        node = Node(parse_url(filename), list())
        assert node.data
        assert node.metadata
        assert len(node.specs) == 2
        assert isinstance(node.specs[0], Label)
        assert isinstance(node.specs[1], Multiscales)


class TestHCSNode:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = tmpdir.mkdir("data")
        self.store = parse_url(str(self.path), mode="w").store
        self.root = zarr.group(store=self.store)

    def test_minimal_plate(self):
        write_plate_metadata(self.root, ["A"], ["1"], ["A/1"])
        row_group = self.root.require_group("A")
        well = row_group.require_group("1")
        write_well_metadata(well, ["0"])
        image = well.require_group("0")
        write_image(zeros((1, 1, 1, 256, 256)), image)

        node = Node(parse_url(str(self.path)), list())
        assert node.data
        assert node.metadata
        assert len(node.specs) == 1
        assert isinstance(node.specs[0], Plate)
        assert node.specs[0].row_names == ["A"]
        assert node.specs[0].col_names == ["1"]
        assert node.specs[0].well_paths == ["A/1"]
        assert node.specs[0].row_count == 1
        assert node.specs[0].column_count == 1

        node = Node(parse_url(str(self.path / "A" / "1")), list())
        assert node.data
        assert node.metadata
        assert len(node.specs) == 1
        assert isinstance(node.specs[0], Well)

    @pytest.mark.parametrize("fmt", (FormatV01(), FormatV02(), FormatV03()))
    def test_multiwells_plate(self, fmt):
        row_names = ["A", "B", "C"]
        col_names = ["1", "2", "3", "4"]
        well_paths = ["A/1", "A/2", "A/4", "B/2", "B/3", "C/1", "C/3", "C/4"]
        write_plate_metadata(self.root, row_names, col_names, well_paths, fmt=fmt)
        for wp in well_paths:
            row, col = wp.split("/")
            row_group = self.root.require_group(row)
            well = row_group.require_group(col)
            write_well_metadata(well, ["0", "1", "2"], fmt=fmt)
            for field in range(3):
                image = well.require_group(str(field))
                write_image(zeros((1, 1, 1, 256, 256)), image)

        node = Node(parse_url(str(self.path)), list())
        assert node.data
        assert node.metadata
        assert len(node.specs) == 1
        assert isinstance(node.specs[0], Plate)
        assert node.specs[0].row_names == row_names
        assert node.specs[0].col_names == col_names
        assert node.specs[0].well_paths == well_paths
        assert node.specs[0].row_count == 3
        assert node.specs[0].column_count == 4

        for wp in well_paths:
            node = Node(parse_url(str(self.path / wp)), list())
            assert node.data
            assert node.metadata
            assert len(node.specs) == 1
            assert isinstance(node.specs[0], Well)

        empty_wells = ["A/3", "B/1", "B/4", "C/2"]
        for wp in empty_wells:
            assert parse_url(str(self.path / wp)) is None

    @pytest.mark.parametrize(
        "axes, dims",
        (
            (["y", "x"], (256, 256)),
            (["t", "y", "x"], (1, 256, 256)),
            (["z", "y", "x"], (1, 256, 256)),
            (["c", "y", "x"], (1, 256, 256)),
            (["c", "z", "y", "x"], (1, 1, 256, 256)),
            (["t", "z", "y", "x"], (1, 1, 256, 256)),
            (["t", "c", "y", "x"], (1, 1, 256, 256)),
        ),
    )
    def test_plate_2D5D(self, axes, dims):
        write_plate_metadata(self.root, ["A"], ["1"], ["A/1"], fmt=FormatV03())
        row_group = self.root.require_group("A")
        well = row_group.require_group("1")
        write_well_metadata(well, ["0"], fmt=FormatV03())
        image = well.require_group("0")
        write_image(zeros(dims), image, fmt=FormatV03(), axes=axes)

        node = Node(parse_url(str(self.path)), list())
        assert node.data
        assert node.metadata
        assert len(node.specs) == 1
        assert isinstance(node.specs[0], Plate)

        node = Node(parse_url(str(self.path / "A" / "1")), list())
        assert node.data
        assert node.metadata
        assert len(node.specs) == 1
        assert isinstance(node.specs[0], Well)
