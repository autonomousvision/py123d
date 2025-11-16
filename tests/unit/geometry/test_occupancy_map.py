import numpy as np
import pytest
import shapely.geometry as geom

from py123d.geometry import OccupancyMap2D


class TestOccupancyMap2D:
    """Unit tests for OccupancyMap2D class."""

    def setup_method(self):
        """Set up test fixtures with various geometries."""
        self.square1 = geom.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        self.square2 = geom.Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])
        self.circle = geom.Point(1, 1).buffer(0.5)
        self.line = geom.LineString([(0, 0), (1, 1)])

        self.geometries = [self.square1, self.square2, self.circle, self.line]
        self.string_ids = ["square1", "square2", "circle", "line"]
        self.int_ids = [1, 2, 3, 4]

    def test_init_with_default_ids(self):
        """Test initialization with default string IDs."""
        occ_map = OccupancyMap2D(self.geometries)

        assert len(occ_map) == 4
        assert occ_map.ids == ["0", "1", "2", "3"]
        assert len(occ_map.geometries) == 4

    def test_init_with_string_ids(self):
        """Test initialization with custom string IDs."""
        occ_map = OccupancyMap2D(self.geometries, ids=self.string_ids)

        assert len(occ_map) == 4
        assert occ_map.ids == self.string_ids
        assert occ_map["square1"] == self.square1

    def test_init_with_int_ids(self):
        """Test initialization with integer IDs."""
        occ_map = OccupancyMap2D(self.geometries, ids=self.int_ids)

        assert len(occ_map) == 4
        assert occ_map.ids == self.int_ids
        assert occ_map[1] == self.square1

    def test_init_with_mismatched_ids_length(self):
        """Test that initialization fails with mismatched IDs length."""
        with pytest.raises(AssertionError):
            OccupancyMap2D(self.geometries, ids=["id1", "id2"])

    def test_init_with_custom_node_capacity(self):
        """Test initialization with custom node capacity."""
        occ_map = OccupancyMap2D(self.geometries, node_capacity=5)
        assert occ_map._node_capacity == 5

    def test_from_dict_constructor(self):
        """Test construction from dictionary."""
        geometry_dict = {"square": self.square1, "circle": self.circle, "line": self.line}

        occ_map = OccupancyMap2D.from_dict(geometry_dict)

        assert len(occ_map) == 3
        assert set(occ_map.ids) == set(["square", "circle", "line"])
        assert occ_map["square"] == self.square1

    def test_getitem_string_id(self):
        """Test geometry retrieval by string ID."""
        occ_map = OccupancyMap2D(self.geometries, ids=self.string_ids)

        assert occ_map["square1"] == self.square1
        assert occ_map["circle"] == self.circle

    def test_getitem_int_id(self):
        """Test geometry retrieval by integer ID."""
        occ_map = OccupancyMap2D(self.geometries, ids=self.int_ids)

        assert occ_map[1] == self.square1
        assert occ_map[3] == self.circle

    def test_getitem_invalid_id(self):
        """Test that invalid ID raises KeyError."""
        occ_map = OccupancyMap2D(self.geometries, ids=self.string_ids)

        with pytest.raises(KeyError):
            _ = occ_map["nonexistent"]

    def test_len(self):
        """Test length property."""
        occ_map = OccupancyMap2D(self.geometries)
        assert len(occ_map) == 4

        empty_map = OccupancyMap2D([])
        assert len(empty_map) == 0

    def test_ids_property(self):
        """Test IDs property getter."""
        occ_map = OccupancyMap2D(self.geometries, ids=self.string_ids)
        assert occ_map.ids == self.string_ids

    def test_geometries_property(self):
        """Test geometries property getter."""
        occ_map = OccupancyMap2D(self.geometries)
        assert list(occ_map.geometries) == self.geometries

    def test_id_to_idx_property(self):
        """Test id_to_idx property."""
        occ_map = OccupancyMap2D(self.geometries, ids=self.string_ids)
        expected_mapping = {"square1": 0, "square2": 1, "circle": 2, "line": 3}
        assert occ_map.id_to_idx == expected_mapping

    def test_intersects_with_overlapping_geometry(self):
        """Test intersects method with overlapping geometry."""
        occ_map = OccupancyMap2D(self.geometries, ids=self.string_ids)

        # Create a geometry that intersects with square1 and circle
        query_geom = geom.Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)])
        intersecting_ids = occ_map.intersects(query_geom)

        # NOTE: square2 does not intersect with the query geometry, the rest does.
        assert "square1" in intersecting_ids
        assert "circle" in intersecting_ids
        assert "line" in intersecting_ids
        assert len(intersecting_ids) == 3

    def test_intersects_with_non_overlapping_geometry(self):
        """Test intersects method with non-overlapping geometry."""
        occ_map = OccupancyMap2D(self.geometries, ids=self.string_ids)

        # Create a geometry that doesn't intersect with any
        query_geom = geom.Polygon([(10, 10), (12, 10), (12, 12), (10, 12)])
        intersecting_ids = occ_map.intersects(query_geom)

        assert len(intersecting_ids) == 0

    def test_query_with_intersects_predicate(self):
        """Test query method with intersects predicate."""
        occ_map = OccupancyMap2D(self.geometries, ids=self.string_ids)

        query_geom = geom.Point(1, 1)
        indices = occ_map.query(query_geom, predicate="intersects")
        assert isinstance(indices, np.ndarray)
        assert occ_map.id_to_idx["square1"] in indices
        assert occ_map.id_to_idx["circle"] in indices
        assert occ_map.id_to_idx["line"] in indices
        assert occ_map.id_to_idx["square2"] not in indices

    def test_query_with_contains_predicate(self):
        """Test query method with contains predicate."""
        occ_map = OccupancyMap2D(self.geometries, ids=self.string_ids)

        query_geom = geom.Point(4, 4)
        indices = occ_map.query(query_geom, predicate="within")

        assert isinstance(indices, np.ndarray)
        assert occ_map.id_to_idx["square2"] in indices
        assert occ_map.id_to_idx["square1"] not in indices
        assert occ_map.id_to_idx["circle"] not in indices
        assert occ_map.id_to_idx["line"] not in indices

    def test_query_with_distance(self):
        """Test query method with distance parameter."""
        occ_map = OccupancyMap2D(self.geometries, ids=self.string_ids)

        query_geom = geom.Point(4, 4)
        indices = occ_map.query(query_geom, predicate="dwithin", distance=3.0)

        assert isinstance(indices, np.ndarray)
        assert occ_map.id_to_idx["square2"] in indices
        assert occ_map.id_to_idx["square1"] in indices
        assert occ_map.id_to_idx["circle"] not in indices
        assert occ_map.id_to_idx["line"] not in indices

    def test_query_nearest_basic(self):
        """Test query_nearest method basic functionality."""
        occ_map = OccupancyMap2D(self.geometries, ids=self.string_ids)

        query_geom = geom.Point(4, 4)
        nearest_indices = occ_map.query_nearest(query_geom)

        assert isinstance(nearest_indices, np.ndarray)

    def test_query_nearest_with_distance(self):
        """Test query_nearest method with return_distance=True."""
        occ_map = OccupancyMap2D(self.geometries)

        query_geom = geom.Point(1, 1)
        result = occ_map.query_nearest(query_geom, return_distance=True)

        assert isinstance(result, tuple)
        assert len(result) == 2
        indices, distances = result
        assert isinstance(indices, np.ndarray)
        assert isinstance(distances, np.ndarray)

    def test_query_nearest_with_max_distance(self):
        """Test query_nearest method with max_distance."""
        occ_map = OccupancyMap2D(self.geometries)

        query_geom = geom.Point(10, 10)

        nearest_indices = occ_map.query_nearest(query_geom, max_distance=1.0)
        assert isinstance(nearest_indices, np.ndarray)
        assert len(nearest_indices) == 0

        nearest_indices = occ_map.query_nearest(query_geom, max_distance=10.0)
        assert isinstance(nearest_indices, np.ndarray)
        assert len(nearest_indices) > 0

    def test_contains_vectorized_single_point(self):
        """Test contains_vectorized with a single point."""
        occ_map = OccupancyMap2D(self.geometries)

        points = np.array([[1.0, 1.0]])  # Point inside square1 and circle
        result = occ_map.contains_vectorized(points)

        assert result.shape == (4, 1)
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool

    def test_contains_vectorized_multiple_points(self):
        """Test contains_vectorized with multiple points."""
        occ_map = OccupancyMap2D(self.geometries)

        points = np.array(
            [
                [1.0, 1.0],  # Inside square1 and circle
                [4.0, 4.0],  # Inside square2
                [10.0, 10.0],  # Outside all geometries
            ]
        )
        result = occ_map.contains_vectorized(points)

        assert result.shape == (4, 3)
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool

        # Check specific containment results
        # Point [1.0, 1.0] should be in square1 (index 0) and circle (index 2)
        assert result[0, 0]  # square1 contains point 0
        assert not result[1, 0]  # square2 does not contain point 0
        assert result[2, 0]  # circle contains point 0
        assert not result[3, 0]  # line does not contain point 0

        # Point [4.0, 4.0] should be in square2 (index 1) only
        assert not result[0, 1]  # square1 does not contain point 1
        assert result[1, 1]  # square2 contains point 1
        assert not result[2, 1]  # circle does not contain point 1
        assert not result[3, 1]  # line does not contain point 1

        # Point [10.0, 10.0] should not be in any geometry
        assert not result[0, 2]  # square1 does not contain point 2
        assert not result[1, 2]  # square2 does not contain point 2
        assert not result[2, 2]  # circle does not contain point 2
        assert not result[3, 2]  # line does not contain point 2

    def test_contains_vectorized_empty_points(self):
        """Test contains_vectorized with empty points array."""
        occ_map = OccupancyMap2D(self.geometries)

        points = np.empty((0, 2))
        result = occ_map.contains_vectorized(points)

        assert result.shape == (4, 0)

    def test_empty_occupancy_map(self):
        """Test behavior with empty geometry list."""
        occ_map = OccupancyMap2D([])

        assert len(occ_map) == 0
        assert occ_map.ids == []
        assert len(occ_map.geometries) == 0

    def test_single_geometry_map(self):
        """Test behavior with single geometry."""
        occ_map = OccupancyMap2D([self.square1], ids=["single"])

        assert len(occ_map) == 1
        assert occ_map.ids == ["single"]
        assert occ_map["single"] == self.square1
