import numpy as np
import numpy.typing as npt
import shapely

from py123d.geometry.geometry_index import (
    BoundingBoxSE3Index,
    Corners2DIndex,
    Corners3DIndex,
    Point2DIndex,
    Point3DIndex,
    PoseSE3Index,
)
from py123d.geometry.pose import PoseSE3
from py123d.geometry.rotation import EulerAngles
from py123d.geometry.transform.transform_se3 import translate_se3_along_body_frame
from py123d.geometry.utils.bounding_box_utils import (
    bbse2_array_to_corners_array,
    bbse2_array_to_polygon_array,
    bbse3_array_to_corners_array,
    corners_2d_array_to_polygon_array,
    corners_array_to_3d_mesh,
    corners_array_to_edge_lines,
    get_corners_3d_factors,
    points_3d_in_bbse3_array,
)
from py123d.geometry.utils.rotation_utils import get_quaternion_array_from_euler_array
from py123d.geometry.vector import Vector3D


class TestBoundingBoxUtils:  # noqa: PLR0904
    def setup_method(self):
        self._num_consistency_checks = 10
        self._max_pose_xyz = 100.0
        self._max_extent = 200.0

    def _get_random_quat_se3_array(self, size: int) -> npt.NDArray[np.float64]:
        """Generate random SE3 poses in quaternion representation."""
        euler_angles = np.zeros((size, 3), dtype=np.float64)
        euler_angles[:, 0] = np.random.uniform(-np.pi, np.pi, size)  # roll
        euler_angles[:, 1] = np.random.uniform(-np.pi / 2, np.pi / 2, size)  # pitch
        euler_angles[:, 2] = np.random.uniform(-np.pi, np.pi, size)  # yaw

        random_se3_array = np.zeros((size, len(PoseSE3Index)), dtype=np.float64)
        random_se3_array[:, PoseSE3Index.XYZ] = np.random.uniform(
            -self._max_pose_xyz,
            self._max_pose_xyz,
            (size, len(Point3DIndex)),
        )
        random_se3_array[:, PoseSE3Index.QUATERNION] = get_quaternion_array_from_euler_array(euler_angles)

        return random_se3_array

    def test_bbse2_array_to_corners_array_one_dim(self):
        bounding_box_se2_array = np.array([1.0, 2.0, 0.0, 4.0, 2.0])
        corners_array = bbse2_array_to_corners_array(bounding_box_se2_array)

        # fill expected
        expected_corners = np.zeros((4, 2), dtype=np.float64)
        expected_corners[Corners2DIndex.FRONT_LEFT] = [1.0 + 2.0, 2.0 + 1.0]
        expected_corners[Corners2DIndex.FRONT_RIGHT] = [1.0 + 2.0, 2.0 - 1.0]
        expected_corners[Corners2DIndex.BACK_RIGHT] = [1.0 - 2.0, 2.0 - 1.0]
        expected_corners[Corners2DIndex.BACK_LEFT] = [1.0 - 2.0, 2.0 + 1.0]

        np.testing.assert_allclose(corners_array, expected_corners, atol=1e-6)

    def test_bbse2_array_to_corners_array_n_dim(self):
        bounding_box_se2_array = np.array([1.0, 2.0, 0.0, 4.0, 2.0])
        bounding_box_se2_array = np.tile(bounding_box_se2_array, (3, 1))

        corners_array = bbse2_array_to_corners_array(bounding_box_se2_array)

        # fill expected
        expected_corners = np.zeros((4, 2), dtype=np.float64)
        expected_corners[Corners2DIndex.FRONT_LEFT] = [1.0 + 2.0, 2.0 + 1.0]
        expected_corners[Corners2DIndex.FRONT_RIGHT] = [1.0 + 2.0, 2.0 - 1.0]
        expected_corners[Corners2DIndex.BACK_RIGHT] = [1.0 - 2.0, 2.0 - 1.0]
        expected_corners[Corners2DIndex.BACK_LEFT] = [1.0 - 2.0, 2.0 + 1.0]
        expected_corners = np.tile(expected_corners, (3, 1, 1))

        np.testing.assert_allclose(corners_array, expected_corners, atol=1e-6)

    def test_bbse2_array_to_corners_array_zero_dim(self):
        bounding_box_se2_array = np.zeros((0, 5), dtype=np.float64)
        corners_array = bbse2_array_to_corners_array(bounding_box_se2_array)
        expected_corners = np.zeros((0, 4, 2), dtype=np.float64)
        np.testing.assert_allclose(corners_array, expected_corners, atol=1e-6)

    def test_bbse2_array_to_corners_array_rotation(self):
        bounding_box_se2_array = np.array([1.0, 2.0, np.pi / 2, 4.0, 2.0])
        corners_array = bbse2_array_to_corners_array(bounding_box_se2_array)

        # fill expected
        expected_corners = np.zeros((len(Corners2DIndex), len(Point2DIndex)), dtype=np.float64)
        expected_corners[Corners2DIndex.FRONT_LEFT] = [1.0 - 1.0, 2.0 + 2.0]
        expected_corners[Corners2DIndex.FRONT_RIGHT] = [1.0 + 1.0, 2.0 + 2.0]
        expected_corners[Corners2DIndex.BACK_RIGHT] = [1.0 + 1.0, 2.0 - 2.0]
        expected_corners[Corners2DIndex.BACK_LEFT] = [1.0 - 1.0, 2.0 - 2.0]

        np.testing.assert_allclose(corners_array, expected_corners, atol=1e-6)

    def test_corners_2d_array_to_polygon_array_one_dim(self):
        corners_array = np.array(
            [
                [3.0, 3.0],
                [3.0, 1.0],
                [-1.0, 1.0],
                [-1.0, 3.0],
            ]
        )
        polygon = corners_2d_array_to_polygon_array(corners_array)

        expected_polygon = shapely.geometry.Polygon(corners_array)
        np.testing.assert_allclose(polygon.area, expected_polygon.area, atol=1e-6)
        assert polygon.equals(expected_polygon)

    def test_corners_2d_array_to_polygon_array_n_dim(self):
        corners_array = np.array(
            [
                [
                    [3.0, 3.0],
                    [3.0, 1.0],
                    [-1.0, 1.0],
                    [-1.0, 3.0],
                ],
                [
                    [4.0, 4.0],
                    [4.0, 2.0],
                    [0.0, 2.0],
                    [0.0, 4.0],
                ],
            ]
        )
        polygons = corners_2d_array_to_polygon_array(corners_array)

        expected_polygon_1 = shapely.geometry.Polygon(corners_array[0])
        expected_polygon_2 = shapely.geometry.Polygon(corners_array[1])

        np.testing.assert_allclose(polygons[0].area, expected_polygon_1.area, atol=1e-6)
        assert polygons[0].equals(expected_polygon_1)

        np.testing.assert_allclose(polygons[1].area, expected_polygon_2.area, atol=1e-6)
        assert polygons[1].equals(expected_polygon_2)

    def test_corners_2d_array_to_polygon_array_zero_dim(self):
        corners_array = np.zeros((0, 4, 2), dtype=np.float64)
        polygons = corners_2d_array_to_polygon_array(corners_array)
        expected_polygons = np.zeros((0,), dtype=np.object_)
        np.testing.assert_array_equal(polygons, expected_polygons)

    def test_bbse2_array_to_polygon_array_one_dim(self):
        bounding_box_se2_array = np.array([1.0, 2.0, 0.0, 4.0, 2.0])
        polygon = bbse2_array_to_polygon_array(bounding_box_se2_array)

        expected_corners = np.zeros((4, 2), dtype=np.float64)
        expected_corners[Corners2DIndex.FRONT_LEFT] = [1.0 + 2.0, 2.0 + 1.0]
        expected_corners[Corners2DIndex.FRONT_RIGHT] = [1.0 + 2.0, 2.0 - 1.0]
        expected_corners[Corners2DIndex.BACK_RIGHT] = [1.0 - 2.0, 2.0 - 1.0]
        expected_corners[Corners2DIndex.BACK_LEFT] = [1.0 - 2.0, 2.0 + 1.0]
        expected_polygon = shapely.geometry.Polygon(expected_corners)

        np.testing.assert_allclose(polygon.area, expected_polygon.area, atol=1e-6)
        assert polygon.equals(expected_polygon)

    def test_bbse2_array_to_polygon_array_n_dim(self):
        bounding_box_se2_array = np.array([1.0, 2.0, 0.0, 4.0, 2.0])
        bounding_box_se2_array = np.tile(bounding_box_se2_array, (3, 1))

        polygons = bbse2_array_to_polygon_array(bounding_box_se2_array)

        expected_corners = np.zeros((4, 2), dtype=np.float64)
        expected_corners[Corners2DIndex.FRONT_LEFT] = [1.0 + 2.0, 2.0 + 1.0]
        expected_corners[Corners2DIndex.FRONT_RIGHT] = [1.0 + 2.0, 2.0 - 1.0]
        expected_corners[Corners2DIndex.BACK_RIGHT] = [1.0 - 2.0, 2.0 - 1.0]
        expected_corners[Corners2DIndex.BACK_LEFT] = [1.0 - 2.0, 2.0 + 1.0]
        expected_polygon = shapely.geometry.Polygon(expected_corners)

        for polygon in polygons:
            np.testing.assert_allclose(polygon.area, expected_polygon.area, atol=1e-6)
            assert polygon.equals(expected_polygon)

    def test_bbse2_array_to_polygon_array_zero_dim(self):
        bounding_box_se2_array = np.zeros((0, 5), dtype=np.float64)
        polygons = bbse2_array_to_polygon_array(bounding_box_se2_array)
        expected_polygons = np.zeros((0,), dtype=np.object_)
        np.testing.assert_array_equal(polygons, expected_polygons)

    def test_bbse3_array_to_corners_array_one_dim(self):
        bounding_box_se3_array = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 4.0, 2.0, 6.0])
        corners_array = bbse3_array_to_corners_array(bounding_box_se3_array)

        # fill expected
        expected_corners = np.zeros((8, 3), dtype=np.float64)
        expected_corners[Corners3DIndex.FRONT_LEFT_BOTTOM] = [1.0 + 2.0, 2.0 + 1.0, 3.0 - 3.0]
        expected_corners[Corners3DIndex.FRONT_RIGHT_BOTTOM] = [1.0 + 2.0, 2.0 - 1.0, 3.0 - 3.0]
        expected_corners[Corners3DIndex.BACK_RIGHT_BOTTOM] = [1.0 - 2.0, 2.0 - 1.0, 3.0 - 3.0]
        expected_corners[Corners3DIndex.BACK_LEFT_BOTTOM] = [1.0 - 2.0, 2.0 + 1.0, 3.0 - 3.0]
        expected_corners[Corners3DIndex.FRONT_LEFT_TOP] = [1.0 + 2.0, 2.0 + 1.0, 3.0 + 3.0]
        expected_corners[Corners3DIndex.FRONT_RIGHT_TOP] = [1.0 + 2.0, 2.0 - 1.0, 3.0 + 3.0]
        expected_corners[Corners3DIndex.BACK_RIGHT_TOP] = [1.0 - 2.0, 2.0 - 1.0, 3.0 + 3.0]
        expected_corners[Corners3DIndex.BACK_LEFT_TOP] = [1.0 - 2.0, 2.0 + 1.0, 3.0 + 3.0]

        np.testing.assert_allclose(corners_array, expected_corners, atol=1e-6)

    def test_bbse3_array_to_corners_array_one_dim_rotation(self):
        for _ in range(self._num_consistency_checks):
            se3_state = PoseSE3.from_array(self._get_random_quat_se3_array(1)[0])
            se3_array = se3_state.array

            # construct a bounding box
            bounding_box_se3_array = np.zeros((len(BoundingBoxSE3Index),), dtype=np.float64)
            length, width, height = np.random.uniform(0.0, self._max_extent, size=3)

            bounding_box_se3_array[BoundingBoxSE3Index.SE3] = se3_array
            bounding_box_se3_array[BoundingBoxSE3Index.LENGTH] = length
            bounding_box_se3_array[BoundingBoxSE3Index.WIDTH] = width
            bounding_box_se3_array[BoundingBoxSE3Index.HEIGHT] = height

            corners_array = bbse3_array_to_corners_array(bounding_box_se3_array)

            corners_3d_factors = get_corners_3d_factors()
            for corner_idx in Corners3DIndex:
                body_translate_vector = Vector3D.from_array(
                    corners_3d_factors[corner_idx] * bounding_box_se3_array[BoundingBoxSE3Index.EXTENT]
                )
                np.testing.assert_allclose(
                    corners_array[corner_idx],
                    translate_se3_along_body_frame(se3_state, body_translate_vector).point_3d.array,
                    atol=1e-6,
                )

    def test_bbse3_array_to_corners_array_n_dim(self):
        for _ in range(self._num_consistency_checks):
            N = np.random.randint(1, 20)
            se3_state_array = self._get_random_quat_se3_array(N)

            # construct a bounding box
            bounding_box_se3_array = np.zeros((N, len(BoundingBoxSE3Index)), dtype=np.float64)
            lengths, widths, heights = np.random.uniform(0.0, self._max_extent, size=(3, N))

            bounding_box_se3_array[:, BoundingBoxSE3Index.SE3] = se3_state_array
            bounding_box_se3_array[:, BoundingBoxSE3Index.LENGTH] = lengths
            bounding_box_se3_array[:, BoundingBoxSE3Index.WIDTH] = widths
            bounding_box_se3_array[:, BoundingBoxSE3Index.HEIGHT] = heights

            corners_array = bbse3_array_to_corners_array(bounding_box_se3_array)

            corners_3d_factors = get_corners_3d_factors()
            for obj_idx in range(N):
                for corner_idx in Corners3DIndex:
                    body_translate_vector = Vector3D.from_array(
                        corners_3d_factors[corner_idx] * bounding_box_se3_array[obj_idx, BoundingBoxSE3Index.EXTENT]
                    )
                    np.testing.assert_allclose(
                        corners_array[obj_idx, corner_idx],
                        translate_se3_along_body_frame(
                            PoseSE3.from_array(bounding_box_se3_array[obj_idx, BoundingBoxSE3Index.SE3]),
                            body_translate_vector,
                        ).point_3d.array,
                        atol=1e-6,
                    )

    def test_bbse3_array_to_corners_array_zero_dim(self):
        bounding_box_se3_array = np.zeros((0, len(BoundingBoxSE3Index)), dtype=np.float64)
        corners_array = bbse3_array_to_corners_array(bounding_box_se3_array)
        expected_corners = np.zeros((0, 8, 3), dtype=np.float64)
        np.testing.assert_allclose(corners_array, expected_corners, atol=1e-6)

    def test_corners_array_to_3d_mesh_one_dim(self):
        """Test conversion of a single bounding box corners to 3D mesh vertices and faces."""

        bounding_box_se3_array = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 4.0, 2.0, 6.0])
        corners_array = bbse3_array_to_corners_array(bounding_box_se3_array)
        vertices, faces = corners_array_to_3d_mesh(corners_array)

        assert vertices.shape == (8, 3)
        assert faces.shape == (12, 3)
        np.testing.assert_allclose(vertices, corners_array, atol=1e-6)

    def test_corners_array_to_3d_mesh_n_dim(self):
        """Test conversion of multiple bounding box corners to 3D mesh vertices and faces."""

        bounding_box_se3_array = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 4.0, 2.0, 6.0])
        bounding_box_se3_array = np.tile(bounding_box_se3_array, (3, 1))
        corners_array = bbse3_array_to_corners_array(bounding_box_se3_array)

        vertices, faces = corners_array_to_3d_mesh(corners_array)
        assert vertices.shape == (3 * 8, 3), f"Vertices shape: {vertices.shape}"
        assert faces.shape == (3 * 12, 3), f"Faces shape: {faces.shape}"

    def test_corners_array_to_3d_mesh_error(self):
        """Test conversion of zero bounding box corners to 3D mesh vertices and faces."""

        corners_array = np.zeros((3), dtype=np.float64)

        with np.testing.assert_raises(AssertionError):
            corners_array_to_3d_mesh(corners_array)

    def test_corners_array_to_edge_lines_one_dim(self):
        """Test conversion of a single bounding box corners to 3D mesh vertices and faces."""

        bounding_box_se3_array = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 4.0, 2.0, 6.0])
        corners_array = bbse3_array_to_corners_array(bounding_box_se3_array)
        edge_lines = corners_array_to_edge_lines(corners_array)

        assert edge_lines.shape == (12, 2, 3)
        expected_num_lines = 12
        assert edge_lines.shape[0] == expected_num_lines, (
            f"Expected {expected_num_lines} edge lines, got {edge_lines.shape[0]}"
        )
        np.testing.assert_allclose(
            edge_lines.reshape(-1, 3),
            corners_array[
                np.array(
                    [
                        Corners3DIndex.FRONT_LEFT_BOTTOM,
                        Corners3DIndex.FRONT_RIGHT_BOTTOM,
                        Corners3DIndex.FRONT_RIGHT_BOTTOM,
                        Corners3DIndex.BACK_RIGHT_BOTTOM,
                        Corners3DIndex.BACK_RIGHT_BOTTOM,
                        Corners3DIndex.BACK_LEFT_BOTTOM,
                        Corners3DIndex.BACK_LEFT_BOTTOM,
                        Corners3DIndex.FRONT_LEFT_BOTTOM,
                        Corners3DIndex.FRONT_LEFT_TOP,
                        Corners3DIndex.FRONT_RIGHT_TOP,
                        Corners3DIndex.FRONT_RIGHT_TOP,
                        Corners3DIndex.BACK_RIGHT_TOP,
                        Corners3DIndex.BACK_RIGHT_TOP,
                        Corners3DIndex.BACK_LEFT_TOP,
                        Corners3DIndex.BACK_LEFT_TOP,
                        Corners3DIndex.FRONT_LEFT_TOP,
                        Corners3DIndex.FRONT_LEFT_BOTTOM,
                        Corners3DIndex.FRONT_LEFT_TOP,
                        Corners3DIndex.FRONT_RIGHT_BOTTOM,
                        Corners3DIndex.FRONT_RIGHT_TOP,
                        Corners3DIndex.BACK_RIGHT_BOTTOM,
                        Corners3DIndex.BACK_RIGHT_TOP,
                        Corners3DIndex.BACK_LEFT_BOTTOM,
                        Corners3DIndex.BACK_LEFT_TOP,
                    ]
                )
            ],
            atol=1e-6,
        )

    def test_corners_array_to_edge_lines_n_dim(self):
        """Test conversion of multiple bounding box corners to 3D mesh vertices and faces."""

        bounding_box_se3_array = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 4.0, 2.0, 6.0])
        bounding_box_se3_array = np.tile(bounding_box_se3_array, (3, 1))
        corners_array = bbse3_array_to_corners_array(bounding_box_se3_array)

        edge_lines = corners_array_to_edge_lines(corners_array)
        assert edge_lines.shape == (3, 12, 2, 3), f"Edge lines shape: {edge_lines.shape}"

    def test_points_3d_in_bbse3_array(self):
        """Test conversion of zero bounding box corners to 3D mesh vertices and faces."""

        bounding_box_se3_array = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

        points_3d = np.array(
            [
                [0.0, 0.0, 0.0],  # True
                [0.25, 0.25, 0.25],  # True
                [-0.25, 0.25, 0.25],  # True
                [1.0, 1.0, 1.0],  # False
                [1.5, 0.5, 0.5],  # False
            ]
        )

        is_interior = points_3d_in_bbse3_array(points_3d, bounding_box_se3_array)

        assert is_interior.shape == (5,)
        expected_is_interior = np.array([True, True, True, False, False])
        np.testing.assert_array_equal(is_interior, expected_is_interior)

    def test_points_3d_in_bbse3_array_rotated(self):
        """Test points_3d_in_bbse3_array with rotated bounding box."""
        # Create a bounding box with 90-degree rotation around z-axis

        quaternion = EulerAngles(0.0, 0.0, np.pi / 2).quaternion

        bounding_box_se3_array = np.array(
            [
                0.0,
                0.0,
                0.0,  # xyz
                quaternion[0],
                quaternion[1],
                quaternion[2],
                quaternion[3],  # quaternion (90-degree rotation around z)
                2.0,
                1.0,
                1.0,  # length, width, height
            ]
        )
        print(bbse3_array_to_corners_array(bounding_box_se3_array))

        points_3d = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.75, 0.0, 0.0],
                [0.0, 0.75, 0.25],
                [2.0, 2.0, 0.0],
            ]
        )

        is_interior = points_3d_in_bbse3_array(points_3d, bounding_box_se3_array)
        expected_is_interior = np.array([True, False, True, False])

        assert is_interior.shape == (4,)
        np.testing.assert_array_equal(is_interior, expected_is_interior)

    def test_points_3d_in_bbse3_array_n_dim(self):
        """Test points_3d_in_bbse3_batch with multiple bounding boxes."""

        # Create multiple bounding boxes
        bounding_box_se3_array = np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [10.0, 10.0, 10.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ]
        )

        # Create test points
        points_3d = np.array(
            [
                [0.0, 0.0, 0.0],  # Inside box 0
                [5.0, 5.0, 5.0],  # Inside box 1
                [10.0, 10.0, 10.0],  # Inside box 2
                [0.5, 0.5, 0.5],  # Inside box 0
                [5.5, 5.5, 5.5],  # Inside box 1
                [100.0, 100.0, 100.0],  # Outside all
            ]
        )

        is_interior = points_3d_in_bbse3_array(points_3d, bounding_box_se3_array)
        is_interior_expected = np.array(
            [
                [True, False, False],
                [False, True, False],
                [False, False, True],
                [True, False, False],
                [False, True, False],
                [False, False, False],
            ]
        ).T
        assert is_interior.shape == (3, 6)
        np.testing.assert_array_equal(is_interior, is_interior_expected)
