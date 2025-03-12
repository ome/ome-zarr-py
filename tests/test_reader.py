import dask.array as da
import numpy as np
import pytest
import zarr
from numpy import ones, zeros

from ome_zarr.data import create_zarr
from ome_zarr.io import parse_url
from ome_zarr.reader import Node, Plate, Reader, Well
from ome_zarr.writer import write_image, write_plate_metadata, write_well_metadata


class TestReader:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = tmpdir.mkdir("data")
        create_zarr(str(self.path))

    def assert_node(self, node: Node):
        if not node.data and not node.metadata:
            assert False, f"Empty node received: {node}"

    def test_image(self):
        reader = Reader(parse_url(str(self.path)))
        assert len(list(reader())) == 3

    def test_labels(self):
        filename = str(self.path.join("labels"))
        reader = Reader(parse_url(filename))
        assert len(list(reader())) == 3

    def test_label(self):
        filename = str(self.path.join("labels", "coins"))
        reader = Reader(parse_url(filename))
        assert len(list(reader())) == 3

    def test_omero(self):
        reader = Reader(parse_url(str(self.path)))()
        image_node = next(iter(reader))
        omero = image_node.zarr.root_attrs.get("omero")
        assert "channels" in omero
        assert isinstance(omero["channels"], list)
        assert len(omero["channels"]) == 1


class TestInvalid:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        self.path = tmpdir.mkdir("data")

    def test_invalid_version(self):
        grp = create_zarr(str(self.path))
        # update version to something invalid
        attrs = grp.attrs.asdict()
        attrs["multiscales"][0]["version"] = "invalid"
        grp.attrs.put(attrs)
        # should raise exception
        with pytest.raises(ValueError) as exe:
            reader = Reader(parse_url(str(self.path)))
            assert len(list(reader())) == 2
        assert str(exe.value) == "Version invalid not recognized"


class TestHCSReader:
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

        reader = Reader(parse_url(str(self.path)))
        nodes = list(reader())
        # currently reading plate labels disabled. Only 1 node
        assert len(nodes) == 1
        assert len(nodes[0].specs) == 1
        assert isinstance(nodes[0].specs[0], Plate)
        # assert len(nodes[1].specs) == 1
        # assert isinstance(nodes[1].specs[0], PlateLabels)

    @pytest.mark.parametrize("field_paths", [["0", "1", "2"], ["img1", "img2", "img3"]])
    def test_multiwells_plate(self, field_paths):
        row_names = ["A", "B", "C"]
        col_names = ["1", "2", "3", "4"]
        well_paths = ["A/1", "A/2", "A/4", "B/2", "B/3", "C/1", "C/3", "C/4"]
        write_plate_metadata(self.root, row_names, col_names, well_paths)
        for wp in well_paths:
            row, col = wp.split("/")
            row_group = self.root.require_group(row)
            well = row_group.require_group(col)
            write_well_metadata(well, field_paths)
            for field in field_paths:
                image = well.require_group(str(field))
                write_image(ones((1, 1, 1, 256, 256)), image)

        reader = Reader(parse_url(str(self.path)))
        nodes = list(reader())
        # currently reading plate labels disabled. Only 1 node
        assert len(nodes) == 1

        plate_node = nodes[0]
        assert len(plate_node.specs) == 1
        assert isinstance(plate_node.specs[0], Plate)
        # data should be a Dask array
        pyramid = plate_node.data
        assert isinstance(pyramid[0], da.Array)
        # if we compute(), expect to get numpy array
        result = pyramid[0].compute()
        assert isinstance(result, np.ndarray)

        # Get the plate node's array. It should be fused from the first field of all
        # well arrays (which in this test are non-zero), with zero values for wells
        # that failed to load (not expected) or the surplus area not filled by a well.
        expected_num_pixels = (
            len(well_paths) * len(field_paths[:1]) * np.prod((1, 1, 1, 256, 256))
        )
        pyramid_0 = pyramid[0]
        assert np.asarray(pyramid_0).sum() == expected_num_pixels

        # assert isinstance(nodes[1].specs[0], PlateLabels)

        reader = Reader(parse_url(f"{self.path}/{well_paths[0]}"))
        nodes = list(reader())
        assert isinstance(nodes[0].specs[0], Well)
        pyramid = nodes[0].data
        assert isinstance(pyramid[0], da.Array)
        result = pyramid[0].compute()
        assert isinstance(result, np.ndarray)
