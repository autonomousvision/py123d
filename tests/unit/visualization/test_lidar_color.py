"""Tests for lidar point-cloud coloring, including semantic / instance segmentation overlays."""

from __future__ import annotations

import numpy as np

from py123d.datatypes import Timestamp
from py123d.datatypes.sensors.lidar import Lidar, LidarFeature, LidarID, LidarMergedMetadata, LidarMetadata
from py123d.parser.lidar_segmentation_registry import WODPerceptionLidarSegmentationLabel
from py123d.visualization.color.default import DEFAULT_LIDAR_SEGMENTATION_COLORS
from py123d.visualization.matplotlib.lidar import get_lidar_pc_color


def _make_lidar(features: dict, num_points: int = 30, segmentation_label_class=None) -> Lidar:
    xyz = np.random.RandomState(0).randn(num_points, 3).astype(np.float32)
    metadata = LidarMergedMetadata(
        {LidarID.LIDAR_TOP: LidarMetadata("top", LidarID.LIDAR_TOP, segmentation_label_class=segmentation_label_class)}
    )
    return Lidar(
        timestamp=Timestamp.from_us(0),
        timestamp_end=Timestamp.from_us(1),
        metadata=metadata,
        point_cloud_3d=xyz,
        point_cloud_features=features,
    )


class TestSemanticColoring:
    def test_returns_rgb_uint8_per_point(self):
        semantic = np.array([0, 1, 2] * 10, dtype=np.uint8)
        lidar = _make_lidar({LidarFeature.SEMANTIC.serialize(): semantic})
        colors = get_lidar_pc_color(lidar, "semantic")
        assert colors.shape == (30, 3)
        assert colors.dtype == np.uint8

    def test_color_is_stable_per_class(self):
        # Same class id must map to the same color regardless of position (frame-to-frame stability).
        semantic = np.array([0, 1, 2, 1, 0], dtype=np.uint8)
        lidar = _make_lidar({LidarFeature.SEMANTIC.serialize(): semantic}, num_points=5)
        colors = get_lidar_pc_color(lidar, "semantic")
        np.testing.assert_array_equal(colors[0], colors[4])  # both class 0
        np.testing.assert_array_equal(colors[1], colors[3])  # both class 1
        assert not np.array_equal(colors[0], colors[1])  # distinct classes -> distinct colors

    def test_color_depends_only_on_id_not_on_present_set(self):
        # A point of class 2 gets the same color whether or not other classes are present.
        a = get_lidar_pc_color(_make_lidar({LidarFeature.SEMANTIC.serialize(): np.array([2], np.uint8)}, 1), "semantic")
        b = get_lidar_pc_color(
            _make_lidar({LidarFeature.SEMANTIC.serialize(): np.array([0, 1, 2, 5], np.uint8)}, 4), "semantic"
        )
        np.testing.assert_array_equal(a[0], b[2])

    def test_missing_feature_falls_back_to_default(self):
        lidar = _make_lidar({LidarFeature.INTENSITY.serialize(): np.zeros(30, np.uint8)})
        colors = get_lidar_pc_color(lidar, "semantic", dark_mode=True)
        assert np.all(colors == 255)  # dark mode default is white


class TestCityscapesSemanticColoring:
    """When the lidar carries its taxonomy, semantic ids get canonical Cityscapes-palette colors."""

    def test_wod_ids_map_to_cityscapes_colors(self):
        wod = WODPerceptionLidarSegmentationLabel
        raw = np.array(
            [wod.TYPE_CAR.value, wod.TYPE_PEDESTRIAN.value, wod.TYPE_ROAD.value, wod.TYPE_UNDEFINED.value],
            dtype=np.uint8,
        )
        lidar = _make_lidar({LidarFeature.SEMANTIC.serialize(): raw}, num_points=4, segmentation_label_class=wod)
        colors = get_lidar_pc_color(lidar, "semantic")
        assert tuple(colors[0]) == (0, 0, 142)  # Cityscapes car
        assert tuple(colors[1]) == (220, 20, 60)  # Cityscapes person
        assert tuple(colors[2]) == (128, 64, 128)  # Cityscapes road
        assert tuple(colors[3]) == (0, 0, 0)  # undefined -> ignore (black)

    def test_color_matches_unified_default_palette(self):
        # The color of a raw id equals the palette color of its to_default() label.
        wod = WODPerceptionLidarSegmentationLabel
        raw = np.array([wod.TYPE_TRUCK.value], dtype=np.uint8)
        lidar = _make_lidar({LidarFeature.SEMANTIC.serialize(): raw}, num_points=1, segmentation_label_class=wod)
        color = get_lidar_pc_color(lidar, "semantic")[0]
        expected = DEFAULT_LIDAR_SEGMENTATION_COLORS[wod.TYPE_TRUCK.to_default()].rgb
        assert tuple(color) == tuple(expected)

    def test_falls_back_to_stable_colors_without_taxonomy(self):
        # No taxonomy on the metadata (e.g. an older log): still colors, just not Cityscapes-keyed.
        raw = np.array([1, 2, 1], dtype=np.uint8)
        lidar = _make_lidar({LidarFeature.SEMANTIC.serialize(): raw}, num_points=3, segmentation_label_class=None)
        colors = get_lidar_pc_color(lidar, "semantic")
        assert colors.shape == (3, 3)
        np.testing.assert_array_equal(colors[0], colors[2])  # stable per id


class TestInstanceColoring:
    def test_instance_returns_rgb_uint8_per_point(self):
        instance = np.arange(30, dtype=np.uint16)
        lidar = _make_lidar({LidarFeature.INSTANCE.serialize(): instance})
        colors = get_lidar_pc_color(lidar, "instance")
        assert colors.shape == (30, 3)
        assert colors.dtype == np.uint8

    def test_instance_missing_feature_falls_back(self):
        lidar = _make_lidar({LidarFeature.INTENSITY.serialize(): np.zeros(30, np.uint8)})
        colors = get_lidar_pc_color(lidar, "instance")
        assert np.all(colors == 0)  # light mode default is black
