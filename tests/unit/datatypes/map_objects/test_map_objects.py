import unittest
from typing import List, Tuple

import numpy as np
import shapely
import trimesh

from py123d.datatypes.map_objects import Intersection, Lane, LaneGroup, MapLayer
from py123d.datatypes.map_objects.map_layer_types import RoadEdgeType, RoadLineType
from py123d.datatypes.map_objects.map_objects import (
    Carpark,
    Crosswalk,
    GenericDrivable,
    RoadEdge,
    RoadLine,
    StopZone,
    Walkway,
)
from py123d.geometry.polyline import Polyline2D, Polyline3D

from .mock_map_api import MockMapAPI


def _get_linked_map_object_setup() -> Tuple[List[Lane], List[LaneGroup], List[Intersection]]:
    """Helper function to create linked map objects for testing."""

    Z = 0.0

    # Lanes:
    lanes: List[Lane] = []

    # Middle Lane 0, group 0
    middle_left_boundary = np.array([[0.0, 1.0, Z], [50.0, 1.0, Z]])
    middle_right_boundary = np.array([[0.0, -1.0, Z], [50.0, -1.0, Z]])
    middle_centerline = np.mean(np.array([middle_right_boundary, middle_left_boundary]), axis=0)
    lanes.append(
        Lane(
            object_id=0,
            lane_group_id=0,
            left_boundary=Polyline3D.from_array(middle_left_boundary),
            right_boundary=Polyline3D.from_array(middle_right_boundary),
            centerline=Polyline3D.from_array(middle_centerline),
            left_lane_id=1,
            right_lane_id=2,
            predecessor_ids=[3],
            successor_ids=[4],
            speed_limit_mps=0.0,
        )
    )

    # Left Lane 1, group 0
    left_left_boundary = np.array([[0.0, 2.0, Z], [50.0, 2.0, Z]])
    left_right_boundary = middle_left_boundary.copy()
    left_centerline = np.mean(np.array([left_right_boundary, left_left_boundary]), axis=0)
    lanes.append(
        Lane(
            object_id=1,
            lane_group_id=0,
            left_boundary=Polyline3D.from_array(left_left_boundary),
            right_boundary=Polyline3D.from_array(left_right_boundary),
            centerline=Polyline3D.from_array(left_centerline),
            left_lane_id=None,
            right_lane_id=0,
            predecessor_ids=[],
            successor_ids=[],
            speed_limit_mps=0.0,
        )
    )

    # Right Lane 2, group 0
    right_right_boundary = np.array([[0.0, -2.0, Z], [50.0, -2.0, Z]])
    right_left_boundary = middle_right_boundary.copy()
    right_centerline = np.mean(np.array([right_right_boundary, right_left_boundary]), axis=0)
    lanes.append(
        Lane(
            object_id=2,
            lane_group_id=0,
            left_boundary=Polyline3D.from_array(right_left_boundary),
            right_boundary=Polyline3D.from_array(right_right_boundary),
            centerline=Polyline3D.from_array(right_centerline),
            left_lane_id=0,
            right_lane_id=None,
            predecessor_ids=[],
            successor_ids=[],
            speed_limit_mps=0.0,
        )
    )

    # Predecessor lane 3, group 1
    predecessor_left_boundary = np.array([[-50.0, 1.0, Z], [0.0, 1.0, Z]])
    predecessor_right_boundary = np.array([[-50.0, -1.0, Z], [0.0, -1.0, Z]])
    predecessor_centerline = np.mean(np.array([predecessor_right_boundary, predecessor_left_boundary]), axis=0)
    lanes.append(
        Lane(
            object_id=3,
            lane_group_id=1,
            left_boundary=Polyline3D.from_array(predecessor_left_boundary),
            right_boundary=Polyline3D.from_array(predecessor_right_boundary),
            centerline=Polyline3D.from_array(predecessor_centerline),
            left_lane_id=None,
            right_lane_id=None,
            predecessor_ids=[],
            successor_ids=[0],
            speed_limit_mps=0.0,
        )
    )

    # Successor lane 4, group 2
    successor_left_boundary = np.array([[50.0, 1.0, Z], [100.0, 1.0, Z]])
    successor_right_boundary = np.array([[50.0, -1.0, Z], [100.0, -1.0, Z]])
    successor_centerline = np.mean(np.array([successor_right_boundary, successor_left_boundary]), axis=0)
    lanes.append(
        Lane(
            object_id=4,
            lane_group_id=2,
            left_boundary=Polyline3D.from_array(successor_left_boundary),
            right_boundary=Polyline3D.from_array(successor_right_boundary),
            centerline=Polyline3D.from_array(successor_centerline),
            left_lane_id=None,
            right_lane_id=None,
            predecessor_ids=[0],
            successor_ids=[],
            speed_limit_mps=0.0,
        )
    )

    # Lane Groups:
    lane_groups = []

    # Middle lane group 0, lanes 0,1,2
    middle_lane_group = LaneGroup(
        object_id=0,
        lane_ids=[0, 1, 2],
        left_boundary=Polyline3D.from_array(left_left_boundary),
        right_boundary=Polyline3D.from_array(left_right_boundary),
        intersection_id=None,
        predecessor_ids=[1],
        successor_ids=[2],
    )
    lane_groups.append(middle_lane_group)

    # Predecessor lane group 1, lane 3, intersection 0
    predecessor_lane_group = LaneGroup(
        object_id=1,
        lane_ids=[3],
        left_boundary=Polyline3D.from_array(predecessor_left_boundary),
        right_boundary=Polyline3D.from_array(predecessor_right_boundary),
        intersection_id=0,
        predecessor_ids=[],
        successor_ids=[0],
    )
    lane_groups.append(predecessor_lane_group)

    # Successor lane group 2, lane 4, intersection 1
    successor_lane_group = LaneGroup(
        object_id=2,
        lane_ids=[4],
        left_boundary=Polyline3D.from_array(successor_left_boundary),
        right_boundary=Polyline3D.from_array(successor_right_boundary),
        intersection_id=1,
        predecessor_ids=[0],
        successor_ids=[],
    )
    lane_groups.append(successor_lane_group)

    # Intersections:
    intersections = []

    # Intersection 0, includes lane groups 1
    intersection_predecessor = Intersection(
        object_id=0,
        lane_group_ids=[1],
        outline=predecessor_lane_group.outline,
    )
    intersections.append(intersection_predecessor)

    intersection_successor = Intersection(
        object_id=1,
        lane_group_ids=[2],
        outline=successor_lane_group.outline,
    )
    intersections.append(intersection_successor)

    return lanes, lane_groups, intersections


class TestLane(unittest.TestCase):
    def setUp(self) -> None:
        lanes, lane_groups, intersections = _get_linked_map_object_setup()
        self.lanes = lanes
        self.lane_groups = lane_groups
        self.intersections = intersections

    def test_set_up(self):
        """Test that the setup function creates the correct number of map objects."""
        self.assertEqual(len(self.lanes), 5)
        self.assertEqual(len(self.lane_groups), 3)
        self.assertEqual(len(self.intersections), 2)

    def test_properties(self):
        """Test that the properties of the Lane objects are correct."""
        lane0 = self.lanes[0]
        self.assertEqual(lane0.layer, MapLayer.LANE)
        self.assertEqual(lane0.lane_group_id, 0)
        self.assertIsInstance(lane0.left_boundary, Polyline3D)
        self.assertIsInstance(lane0.right_boundary, Polyline3D)
        self.assertIsInstance(lane0.centerline, Polyline3D)

        self.assertEqual(lane0.left_lane_id, 1)
        self.assertEqual(lane0.right_lane_id, 2)
        self.assertEqual(lane0.predecessor_ids, [3])
        self.assertEqual(lane0.successor_ids, [4])
        self.assertEqual(lane0.speed_limit_mps, 0.0)
        self.assertIsInstance(lane0.trimesh_mesh, trimesh.base.Trimesh)

    def test_base_properties(self):
        """Test that the base_surface property of the Lane objects is correct."""
        lane0 = self.lanes[0]
        self.assertEqual(lane0.object_id, 0)
        self.assertIsInstance(lane0.outline, Polyline3D)
        self.assertIsInstance(lane0.outline_2d, Polyline2D)
        self.assertIsInstance(lane0.outline_3d, Polyline3D)
        self.assertIsInstance(lane0.shapely_polygon, shapely.Polygon)

    def test_left_links(self):
        """Test that the left neighboring lanes are correctly linked."""
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=True,
        )

        def _no_left_neighbor(lane: Lane):
            self.assertIsNotNone(lane)
            self.assertIsNone(lane.left_lane)
            self.assertIsNone(lane.left_lane_id)

        # Middle Lane 0
        lane0: Lane = map_api.get_map_object(0, MapLayer.LANE)
        self.assertIsNotNone(lane0)
        self.assertIsNotNone(lane0.left_lane)
        self.assertIsInstance(lane0.left_lane, Lane)
        self.assertEqual(lane0.left_lane.object_id, 1)
        self.assertEqual(lane0.left_lane.object_id, lane0.left_lane_id)

        # Left Lane 1
        lane1: Lane = map_api.get_map_object(1, MapLayer.LANE)
        _no_left_neighbor(lane1)

        # Right Lane 2
        lane2: Lane = map_api.get_map_object(2, MapLayer.LANE)
        self.assertIsNotNone(lane2)
        self.assertIsNotNone(lane2.left_lane)
        self.assertIsInstance(lane2.left_lane, Lane)
        self.assertEqual(lane2.left_lane.object_id, 0)
        self.assertEqual(lane2.left_lane.object_id, lane2.left_lane_id)

        # Predecessor Lane 3
        lane3: Lane = map_api.get_map_object(3, MapLayer.LANE)
        _no_left_neighbor(lane3)

        # Successor Lane 4
        lane4: Lane = map_api.get_map_object(4, MapLayer.LANE)
        _no_left_neighbor(lane4)

    def test_right_links(self):
        """Test that the right neighboring lanes are correctly linked."""
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=True,
        )

        def _no_right_neighbor(lane: Lane):
            self.assertIsNotNone(lane)
            self.assertIsNone(lane.right_lane)
            self.assertIsNone(lane.right_lane_id)

        # Middle Lane 0
        lane0: Lane = map_api.get_map_object(0, MapLayer.LANE)
        self.assertIsNotNone(lane0)
        self.assertIsNotNone(lane0.right_lane)
        self.assertIsInstance(lane0.right_lane, Lane)
        self.assertEqual(lane0.right_lane.object_id, 2)
        self.assertEqual(lane0.right_lane.object_id, lane0.right_lane_id)

        # Left Lane 1
        lane1: Lane = map_api.get_map_object(1, MapLayer.LANE)
        self.assertIsNotNone(lane1)
        self.assertIsNotNone(lane1.right_lane)
        self.assertIsInstance(lane1.right_lane, Lane)
        self.assertEqual(lane1.right_lane.object_id, 0)
        self.assertEqual(lane1.right_lane.object_id, lane1.right_lane_id)

        # Right Lane 2
        lane2: Lane = map_api.get_map_object(2, MapLayer.LANE)
        _no_right_neighbor(lane2)

        # Predecessor Lane 3
        lane3: Lane = map_api.get_map_object(3, MapLayer.LANE)
        _no_right_neighbor(lane3)

        # Successor Lane 4
        lane4: Lane = map_api.get_map_object(4, MapLayer.LANE)
        _no_right_neighbor(lane4)

    def test_predecessor_links(self):
        """Test that the predecessor lanes are correctly linked."""
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=True,
        )

        def _no_predecessors(lane: Lane):
            self.assertIsNotNone(lane)
            self.assertEqual(lane.predecessors, [])
            self.assertEqual(lane.predecessor_ids, [])

        # Middle Lane 0
        lane0: Lane = map_api.get_map_object(0, MapLayer.LANE)
        self.assertIsNotNone(lane0)
        self.assertIsNotNone(lane0.predecessors)
        self.assertEqual(len(lane0.predecessors), 1)
        self.assertIsInstance(lane0.predecessors[0], Lane)
        self.assertEqual(lane0.predecessors[0].object_id, 3)
        self.assertEqual(lane0.predecessor_ids, [3])

        # Left Lane 1
        lane1: Lane = map_api.get_map_object(1, MapLayer.LANE)
        _no_predecessors(lane1)

        # Right Lane 2
        lane2: Lane = map_api.get_map_object(2, MapLayer.LANE)
        _no_predecessors(lane2)

        # Predecessor Lane 3
        lane3: Lane = map_api.get_map_object(3, MapLayer.LANE)
        _no_predecessors(lane3)

        # Successor Lane 4
        lane4: Lane = map_api.get_map_object(4, MapLayer.LANE)
        self.assertIsNotNone(lane4)
        self.assertIsNotNone(lane4.predecessors)
        self.assertEqual(len(lane4.predecessors), 1)
        self.assertIsInstance(lane4.predecessors[0], Lane)
        self.assertEqual(lane4.predecessors[0].object_id, 0)
        self.assertEqual(lane4.predecessor_ids, [0])

    def test_successor_links(self):
        """Test that the successor lanes are correctly linked."""
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=True,
        )

        def _no_successors(lane: Lane):
            self.assertIsNotNone(lane)
            self.assertEqual(lane.successors, [])
            self.assertEqual(lane.successor_ids, [])

        # Middle Lane 0
        lane0: Lane = map_api.get_map_object(0, MapLayer.LANE)
        self.assertIsNotNone(lane0)
        self.assertIsNotNone(lane0.successors)
        self.assertEqual(len(lane0.successors), 1)
        self.assertIsInstance(lane0.successors[0], Lane)
        self.assertEqual(lane0.successors[0].object_id, 4)
        self.assertEqual(lane0.successor_ids, [4])

        # Left Lane 1
        lane1: Lane = map_api.get_map_object(1, MapLayer.LANE)
        _no_successors(lane1)

        # Right Lane 2
        lane2: Lane = map_api.get_map_object(2, MapLayer.LANE)
        _no_successors(lane2)

        # Predecessor Lane 3
        lane3: Lane = map_api.get_map_object(3, MapLayer.LANE)
        self.assertIsNotNone(lane3)
        self.assertIsNotNone(lane3.successors)
        self.assertEqual(len(lane3.successors), 1)
        self.assertIsInstance(lane3.successors[0], Lane)
        self.assertEqual(lane3.successors[0].object_id, 0)
        self.assertEqual(lane3.successor_ids, [0])

        # Successor Lane 4
        lane4: Lane = map_api.get_map_object(4, MapLayer.LANE)
        _no_successors(lane4)

    def test_no_links(self):
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=False,
        )
        for lane in self.lanes:
            lane_from_api: Lane = map_api.get_map_object(lane.object_id, MapLayer.LANE)
            self.assertIsNotNone(lane_from_api)
            self.assertIsNone(lane_from_api.left_lane)
            self.assertIsNone(lane_from_api.right_lane)
            self.assertIsNone(lane_from_api.predecessors)
            self.assertIsNone(lane_from_api.successors)

    def test_lane_group_links(self):
        """Test that the lane group links are correct."""
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=True,
        )

        for lane in self.lanes:
            lane_from_api: Lane = map_api.get_map_object(lane.object_id, MapLayer.LANE)
            self.assertIsNotNone(lane_from_api)
            self.assertIsNotNone(lane_from_api.lane_group)
            self.assertIsInstance(lane_from_api.lane_group, LaneGroup)
            self.assertEqual(lane_from_api.lane_group.object_id, lane_from_api.lane_group_id)


class TestLaneGroup(unittest.TestCase):

    def setUp(self):
        lanes, lane_groups, intersections = _get_linked_map_object_setup()
        self.lanes = lanes
        self.lane_groups = lane_groups
        self.intersections = intersections

    def test_properties(self):
        """Test that the properties of the LaneGroup objects are correct."""
        lane_group0 = self.lane_groups[0]
        self.assertEqual(lane_group0.layer, MapLayer.LANE_GROUP)
        self.assertEqual(lane_group0.lane_ids, [0, 1, 2])
        self.assertIsInstance(lane_group0.left_boundary, Polyline3D)
        self.assertIsInstance(lane_group0.right_boundary, Polyline3D)
        self.assertEqual(lane_group0.intersection_id, None)
        self.assertEqual(lane_group0.predecessor_ids, [1])
        self.assertEqual(lane_group0.successor_ids, [2])
        self.assertIsInstance(lane_group0.trimesh_mesh, trimesh.base.Trimesh)

    def test_base_properties(self):
        """Test that the base surface properties of the LaneGroup objects are correct."""
        lane_group0 = self.lane_groups[0]
        self.assertEqual(lane_group0.object_id, 0)
        self.assertIsInstance(lane_group0.outline, Polyline3D)
        self.assertIsInstance(lane_group0.outline_2d, Polyline2D)
        self.assertIsInstance(lane_group0.outline_3d, Polyline3D)
        self.assertIsInstance(lane_group0.shapely_polygon, shapely.Polygon)

    def test_lane_links(self):
        """Test that the lanes are correctly linked to the lane group."""
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=True,
        )

        # Lane group 0 contains lanes 0, 1, 2
        lane_group0: LaneGroup = map_api.get_map_object(0, MapLayer.LANE_GROUP)
        self.assertIsNotNone(lane_group0)
        self.assertIsNotNone(lane_group0.lanes)
        self.assertEqual(len(lane_group0.lanes), 3)
        for i, lane in enumerate(lane_group0.lanes):
            self.assertIsInstance(lane, Lane)
            self.assertEqual(lane.object_id, i)

        # Lane group 1 contains lane 3
        lane_group1: LaneGroup = map_api.get_map_object(1, MapLayer.LANE_GROUP)
        self.assertIsNotNone(lane_group1)
        self.assertIsNotNone(lane_group1.lanes)
        self.assertEqual(len(lane_group1.lanes), 1)
        self.assertIsInstance(lane_group1.lanes[0], Lane)
        self.assertEqual(lane_group1.lanes[0].object_id, 3)

        # Lane group 2 contains lane 4
        lane_group2: LaneGroup = map_api.get_map_object(2, MapLayer.LANE_GROUP)
        self.assertIsNotNone(lane_group2)
        self.assertIsNotNone(lane_group2.lanes)
        self.assertEqual(len(lane_group2.lanes), 1)
        self.assertIsInstance(lane_group2.lanes[0], Lane)
        self.assertEqual(lane_group2.lanes[0].object_id, 4)

    def test_predecessor_links(self):
        """Test that the predecessor lane groups are correctly linked."""
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=True,
        )

        def _no_predecessors(lane_group: LaneGroup):
            self.assertIsNotNone(lane_group)
            self.assertEqual(lane_group.predecessors, [])
            self.assertEqual(lane_group.predecessor_ids, [])

        # Lane group 0 has predecessor lane group 1
        lane_group0: LaneGroup = map_api.get_map_object(0, MapLayer.LANE_GROUP)
        self.assertIsNotNone(lane_group0)
        self.assertIsNotNone(lane_group0.predecessors)
        self.assertEqual(len(lane_group0.predecessors), 1)
        self.assertIsInstance(lane_group0.predecessors[0], LaneGroup)
        self.assertEqual(lane_group0.predecessors[0].object_id, 1)
        self.assertEqual(lane_group0.predecessor_ids, [1])

        # Lane group 1 has no predecessors
        lane_group1: LaneGroup = map_api.get_map_object(1, MapLayer.LANE_GROUP)
        _no_predecessors(lane_group1)

        # Lane group 2 has predecessor lane group 0
        lane_group2: LaneGroup = map_api.get_map_object(2, MapLayer.LANE_GROUP)
        self.assertIsNotNone(lane_group2)
        self.assertIsNotNone(lane_group2.predecessors)
        self.assertEqual(len(lane_group2.predecessors), 1)
        self.assertIsInstance(lane_group2.predecessors[0], LaneGroup)
        self.assertEqual(lane_group2.predecessors[0].object_id, 0)
        self.assertEqual(lane_group2.predecessor_ids, [0])

    def test_successor_links(self):
        """Test that the successor lane groups are correctly linked."""
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=True,
        )

        def _no_successors(lane_group: LaneGroup):
            self.assertIsNotNone(lane_group)
            self.assertEqual(lane_group.successors, [])
            self.assertEqual(lane_group.successor_ids, [])

        # Lane group 0 has successor lane group 2
        lane_group0: LaneGroup = map_api.get_map_object(0, MapLayer.LANE_GROUP)
        self.assertIsNotNone(lane_group0)
        self.assertIsNotNone(lane_group0.successors)
        self.assertEqual(len(lane_group0.successors), 1)
        self.assertIsInstance(lane_group0.successors[0], LaneGroup)
        self.assertEqual(lane_group0.successors[0].object_id, 2)
        self.assertEqual(lane_group0.successor_ids, [2])

        # Lane group 1 has successor lane group 0
        lane_group1: LaneGroup = map_api.get_map_object(1, MapLayer.LANE_GROUP)
        self.assertIsNotNone(lane_group1)
        self.assertIsNotNone(lane_group1.successors)
        self.assertEqual(len(lane_group1.successors), 1)
        self.assertIsInstance(lane_group1.successors[0], LaneGroup)
        self.assertEqual(lane_group1.successors[0].object_id, 0)
        self.assertEqual(lane_group1.successor_ids, [0])

        # Lane group 2 has no successors
        lane_group2: LaneGroup = map_api.get_map_object(2, MapLayer.LANE_GROUP)
        _no_successors(lane_group2)

    def test_intersection_links(self):
        """Test that the intersection links are correct."""
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=True,
        )

        # Lane group 0 has no intersection
        lane_group0: LaneGroup = map_api.get_map_object(0, MapLayer.LANE_GROUP)
        self.assertIsNotNone(lane_group0)
        self.assertIsNone(lane_group0.intersection_id)
        self.assertIsNone(lane_group0.intersection)

        # Lane group 1 has intersection 0
        lane_group1: LaneGroup = map_api.get_map_object(1, MapLayer.LANE_GROUP)
        self.assertIsNotNone(lane_group1)
        self.assertEqual(lane_group1.intersection_id, 0)
        self.assertIsNotNone(lane_group1.intersection)
        self.assertIsInstance(lane_group1.intersection, Intersection)
        self.assertEqual(lane_group1.intersection.object_id, 0)

        # Lane group 2 has intersection 1
        lane_group2: LaneGroup = map_api.get_map_object(2, MapLayer.LANE_GROUP)
        self.assertIsNotNone(lane_group2)
        self.assertEqual(lane_group2.intersection_id, 1)
        self.assertIsNotNone(lane_group2.intersection)
        self.assertIsInstance(lane_group2.intersection, Intersection)
        self.assertEqual(lane_group2.intersection.object_id, 1)

    def test_no_links(self):
        """Test that when map_api is not provided, no links are available."""
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=False,
        )
        for lane_group in self.lane_groups:
            lg_from_api: LaneGroup = map_api.get_map_object(lane_group.object_id, MapLayer.LANE_GROUP)
            self.assertIsNotNone(lg_from_api)
            self.assertIsNone(lg_from_api.lanes)
            self.assertIsNone(lg_from_api.predecessors)
            self.assertIsNone(lg_from_api.successors)
            self.assertIsNone(lg_from_api.intersection)


class TestIntersection(unittest.TestCase):

    def setUp(self):
        lanes, lane_groups, intersections = _get_linked_map_object_setup()
        self.lanes = lanes
        self.lane_groups = lane_groups
        self.intersections = intersections

    def test_properties(self):
        """Test that the properties of the Intersection objects are correct."""
        intersection0 = self.intersections[0]
        self.assertEqual(intersection0.layer, MapLayer.INTERSECTION)
        self.assertEqual(intersection0.lane_group_ids, [1])
        self.assertIsInstance(intersection0.outline, Polyline3D)

    def test_base_properties(self):
        """Test that the base surface properties of the Intersection objects are correct."""
        intersection0 = self.intersections[0]
        self.assertEqual(intersection0.object_id, 0)
        self.assertIsInstance(intersection0.outline, Polyline3D)
        self.assertIsInstance(intersection0.outline_2d, Polyline2D)
        self.assertIsInstance(intersection0.outline_3d, Polyline3D)
        self.assertIsInstance(intersection0.shapely_polygon, shapely.Polygon)

    def test_lane_group_links(self):
        """Test that the lane groups are correctly linked to the intersection."""
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=True,
        )

        # Intersection 0 contains lane group 1
        intersection0: Intersection = map_api.get_map_object(0, MapLayer.INTERSECTION)
        self.assertIsNotNone(intersection0)
        self.assertIsNotNone(intersection0.lane_groups)
        self.assertEqual(len(intersection0.lane_groups), 1)
        self.assertIsInstance(intersection0.lane_groups[0], LaneGroup)
        self.assertEqual(intersection0.lane_groups[0].object_id, 1)

        # Intersection 1 contains lane group 2
        intersection1: Intersection = map_api.get_map_object(1, MapLayer.INTERSECTION)
        self.assertIsNotNone(intersection1)
        self.assertIsNotNone(intersection1.lane_groups)
        self.assertEqual(len(intersection1.lane_groups), 1)
        self.assertIsInstance(intersection1.lane_groups[0], LaneGroup)
        self.assertEqual(intersection1.lane_groups[0].object_id, 2)

    def test_no_links(self):
        """Test that when map_api is not provided, no links are available."""
        map_api = MockMapAPI(
            lanes=self.lanes,
            lane_groups=self.lane_groups,
            intersections=self.intersections,
            add_map_api_links=False,
        )
        for intersection in self.intersections:
            int_from_api: Intersection = map_api.get_map_object(intersection.object_id, MapLayer.INTERSECTION)
            self.assertIsNotNone(int_from_api)
            self.assertIsNone(int_from_api.lane_groups)


class TestCrosswalk(unittest.TestCase):
    def test_properties(self):
        """Test that the properties of the Crosswalk object are correct."""
        outline = Polyline3D.from_array(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        )
        crosswalk = Crosswalk(object_id=0, outline=outline)
        self.assertEqual(crosswalk.layer, MapLayer.CROSSWALK)
        self.assertEqual(crosswalk.object_id, 0)
        self.assertIsInstance(crosswalk.outline, Polyline3D)
        self.assertIsInstance(crosswalk.shapely_polygon, shapely.Polygon)

    def test_init_with_shapely_polygon(self):
        """Test initialization with shapely polygon."""
        shapely_polygon = shapely.Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        crosswalk = Crosswalk(object_id=0, shapely_polygon=shapely_polygon)
        self.assertEqual(crosswalk.object_id, 0)
        self.assertIsInstance(crosswalk.shapely_polygon, shapely.Polygon)
        self.assertIsInstance(crosswalk.outline_2d, Polyline2D)

    def test_init_with_polyline2d(self):
        """Test initialization with Polyline2D outline."""
        outline = Polyline2D.from_array(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]))
        crosswalk = Crosswalk(object_id=0, outline=outline)
        self.assertIsInstance(crosswalk.outline_2d, Polyline2D)
        self.assertIsInstance(crosswalk.shapely_polygon, shapely.Polygon)

    def test_base_surface_properties(self):
        """Test base surface object properties."""
        outline = Polyline3D.from_array(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        )
        crosswalk = Crosswalk(object_id=0, outline=outline)
        self.assertIsInstance(crosswalk.outline_3d, Polyline3D)
        self.assertTrue(crosswalk.shapely_polygon.is_valid)


class TestCarpark(unittest.TestCase):
    def test_properties(self):
        """Test that the properties of the Carpark object are correct."""
        outline = Polyline3D.from_array(
            np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
        )
        carpark = Carpark(object_id=1, outline=outline)
        self.assertEqual(carpark.layer, MapLayer.CARPARK)
        self.assertEqual(carpark.object_id, 1)
        self.assertIsInstance(carpark.outline, Polyline3D)
        self.assertIsInstance(carpark.shapely_polygon, shapely.Polygon)

    def test_init_with_shapely_polygon(self):
        """Test initialization with shapely polygon."""
        shapely_polygon = shapely.Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
        carpark = Carpark(object_id=1, shapely_polygon=shapely_polygon)
        self.assertEqual(carpark.object_id, 1)
        self.assertIsInstance(carpark.shapely_polygon, shapely.Polygon)
        self.assertIsInstance(carpark.outline_2d, Polyline2D)

    def test_init_with_polyline2d(self):
        """Test initialization with Polyline2D outline."""
        outline = Polyline2D.from_array(np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]]))
        carpark = Carpark(object_id=1, outline=outline)
        self.assertIsInstance(carpark.outline_2d, Polyline2D)
        self.assertIsInstance(carpark.shapely_polygon, shapely.Polygon)

    def test_polygon_area(self):
        """Test that the polygon area is calculated correctly."""
        outline = Polyline3D.from_array(
            np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
        )
        carpark = Carpark(object_id=1, outline=outline)
        self.assertAlmostEqual(carpark.shapely_polygon.area, 4.0)


class TestWalkway(unittest.TestCase):
    def test_properties(self):
        """Test that the properties of the Walkway object are correct."""
        outline = Polyline2D.from_array(np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [0.0, 1.0], [0.0, 0.0]]))
        walkway = Walkway(object_id=2, outline=outline)
        self.assertEqual(walkway.layer, MapLayer.WALKWAY)
        self.assertEqual(walkway.object_id, 2)
        self.assertIsInstance(walkway.outline_2d, Polyline2D)
        self.assertIsInstance(walkway.shapely_polygon, shapely.Polygon)

    def test_init_with_shapely_polygon(self):
        """Test initialization with shapely polygon."""
        shapely_polygon = shapely.Polygon([(0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (0.0, 1.0)])
        walkway = Walkway(object_id=2, shapely_polygon=shapely_polygon)
        self.assertEqual(walkway.object_id, 2)
        self.assertIsInstance(walkway.shapely_polygon, shapely.Polygon)

    def test_init_with_polyline3d(self):
        """Test initialization with Polyline3D outline."""
        outline = Polyline3D.from_array(
            np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        )
        walkway = Walkway(object_id=2, outline=outline)
        self.assertIsInstance(walkway.outline_3d, Polyline3D)
        self.assertIsInstance(walkway.shapely_polygon, shapely.Polygon)

    def test_polygon_bounds(self):
        """Test that polygon bounds are correct."""
        outline = Polyline2D.from_array(np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [0.0, 1.0], [0.0, 0.0]]))
        walkway = Walkway(object_id=2, outline=outline)
        bounds = walkway.shapely_polygon.bounds
        self.assertEqual(bounds, (0.0, 0.0, 3.0, 1.0))


class TestGenericDrivable(unittest.TestCase):
    def test_properties(self):
        """Test that the properties of the GenericDrivable object are correct."""
        outline = Polyline3D.from_array(
            np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 3.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]])
        )
        generic_drivable = GenericDrivable(object_id=3, outline=outline)
        self.assertEqual(generic_drivable.layer, MapLayer.GENERIC_DRIVABLE)
        self.assertEqual(generic_drivable.object_id, 3)
        self.assertIsInstance(generic_drivable.outline, Polyline3D)
        self.assertIsInstance(generic_drivable.shapely_polygon, shapely.Polygon)

    def test_init_with_shapely_polygon(self):
        """Test initialization with shapely polygon."""
        shapely_polygon = shapely.Polygon([(0.0, 0.0), (5.0, 0.0), (5.0, 3.0), (0.0, 3.0)])
        generic_drivable = GenericDrivable(object_id=3, shapely_polygon=shapely_polygon)
        self.assertEqual(generic_drivable.object_id, 3)
        self.assertIsInstance(generic_drivable.shapely_polygon, shapely.Polygon)

    def test_init_with_polyline2d(self):
        """Test initialization with Polyline2D outline."""
        outline = Polyline2D.from_array(np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 3.0], [0.0, 3.0], [0.0, 0.0]]))
        generic_drivable = GenericDrivable(object_id=3, outline=outline)
        self.assertIsInstance(generic_drivable.outline_2d, Polyline2D)
        self.assertIsInstance(generic_drivable.shapely_polygon, shapely.Polygon)

    def test_polygon_area(self):
        """Test that the polygon area is calculated correctly."""
        outline = Polyline3D.from_array(
            np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 3.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]])
        )
        generic_drivable = GenericDrivable(object_id=3, outline=outline)
        self.assertAlmostEqual(generic_drivable.shapely_polygon.area, 15.0)


class TestStopZone(unittest.TestCase):
    def test_properties(self):
        """Test that the properties of the StopZone object are correct."""
        shapely_polygon = shapely.Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)])
        stop_zone = StopZone(object_id=4, shapely_polygon=shapely_polygon)
        self.assertEqual(stop_zone.layer, MapLayer.STOP_ZONE)
        self.assertEqual(stop_zone.object_id, 4)
        self.assertIsInstance(stop_zone.shapely_polygon, shapely.Polygon)
        self.assertIsInstance(stop_zone.outline_2d, Polyline2D)

    def test_init_with_polyline3d(self):
        """Test initialization with Polyline3D outline."""
        outline = Polyline3D.from_array(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]])
        )
        stop_zone = StopZone(object_id=4, outline=outline)
        self.assertIsInstance(stop_zone.outline, Polyline3D)
        self.assertIsInstance(stop_zone.shapely_polygon, shapely.Polygon)

    def test_init_with_polyline2d(self):
        """Test initialization with Polyline2D outline."""
        outline = Polyline2D.from_array(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5], [0.0, 0.0]]))
        stop_zone = StopZone(object_id=4, outline=outline)
        self.assertIsInstance(stop_zone.outline_2d, Polyline2D)
        self.assertIsInstance(stop_zone.shapely_polygon, shapely.Polygon)

    def test_polygon_area(self):
        """Test that the polygon area is calculated correctly."""
        shapely_polygon = shapely.Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)])
        stop_zone = StopZone(object_id=4, shapely_polygon=shapely_polygon)
        self.assertAlmostEqual(stop_zone.shapely_polygon.area, 0.5)


class TestRoadEdge(unittest.TestCase):
    def test_properties(self):
        """Test that the properties of the RoadEdge object are correct."""
        polyline = Polyline3D.from_array(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]))
        road_edge = RoadEdge(object_id=5, road_edge_type=1, polyline=polyline)
        self.assertEqual(road_edge.layer, MapLayer.ROAD_EDGE)
        self.assertEqual(road_edge.object_id, 5)
        self.assertEqual(road_edge.road_edge_type, 1)
        self.assertIsInstance(road_edge.polyline, Polyline3D)

    def test_init_with_polyline2d(self):
        """Test initialization with Polyline2D."""
        polyline = Polyline2D.from_array(np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]))
        road_edge = RoadEdge(object_id=5, road_edge_type=1, polyline=polyline)
        self.assertIsInstance(road_edge.polyline, Polyline2D)
        self.assertEqual(road_edge.road_edge_type, 1)

    def test_polyline_length(self):
        """Test that the polyline has correct number of points."""
        polyline = Polyline3D.from_array(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]))
        road_edge = RoadEdge(object_id=5, road_edge_type=1, polyline=polyline)
        self.assertEqual(len(road_edge.polyline.array), 3)

    def test_different_road_edge_types(self):
        """Test different road edge types."""
        polyline = Polyline3D.from_array(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
        for edge_type in RoadEdgeType:
            road_edge = RoadEdge(object_id=5, road_edge_type=edge_type, polyline=polyline)
            self.assertEqual(road_edge.road_edge_type, edge_type)


class TestRoadLine(unittest.TestCase):
    def test_properties(self):
        """Test that the properties of the RoadLine object are correct."""
        polyline = Polyline2D.from_array(np.array([[0.0, 1.0], [10.0, 1.0], [20.0, 1.0]]))
        road_line = RoadLine(object_id=6, road_line_type=2, polyline=polyline)
        self.assertEqual(road_line.layer, MapLayer.ROAD_LINE)
        self.assertEqual(road_line.object_id, 6)
        self.assertEqual(road_line.road_line_type, 2)
        self.assertIsInstance(road_line.polyline, Polyline2D)

    def test_init_with_polyline3d(self):
        """Test initialization with Polyline3D."""
        polyline = Polyline3D.from_array(np.array([[0.0, 1.0, 0.0], [10.0, 1.0, 0.0], [20.0, 1.0, 0.0]]))
        road_line = RoadLine(object_id=6, road_line_type=2, polyline=polyline)
        self.assertIsInstance(road_line.polyline, Polyline3D)
        self.assertEqual(road_line.road_line_type, 2)

    def test_polyline_length(self):
        """Test that the polyline has correct number of points."""
        polyline = Polyline2D.from_array(np.array([[0.0, 1.0], [10.0, 1.0], [20.0, 1.0], [30.0, 1.0]]))
        road_line = RoadLine(object_id=6, road_line_type=2, polyline=polyline)
        self.assertEqual(len(road_line.polyline.array), 4)

    def test_different_road_line_types(self):
        """Test different road line types."""
        polyline = Polyline2D.from_array(np.array([[0.0, 1.0], [10.0, 1.0]]))
        for line_type in RoadLineType:
            road_line = RoadLine(object_id=6, road_line_type=line_type, polyline=polyline)
            self.assertEqual(road_line.road_line_type, line_type)
