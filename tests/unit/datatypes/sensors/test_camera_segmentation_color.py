"""Tests for per-pixel camera semantic segmentation coloring (Cityscapes palette) and ``Camera.rgb_image``."""

from __future__ import annotations

import numpy as np

from py123d.datatypes import Timestamp
from py123d.datatypes.sensors.base_camera import Camera, CameraChannelType, CameraID
from py123d.datatypes.sensors.camera_segmentation_label import (
    DEFAULT_CAMERA_SEGMENTATION_RGB,
    TABLEAU_20_RGB,
    DefaultCameraSegmentationLabel,
    colorize_instance_label_map,
    colorize_semantic_label_map,
)
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraMetadata, PinholeDistortion, PinholeIntrinsics
from py123d.datatypes.sensors.segmentation_camera import SegmentationCameraMetadata
from py123d.geometry.pose import PoseSE3
from py123d.parser.camera_segmentation_registry import (
    Kitti360CameraSegmentationLabel,
    WODPerceptionCameraSegmentationLabel,
)

# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------


def _make_rgb_camera_metadata(width: int = 8, height: int = 6) -> PinholeCameraMetadata:
    return PinholeCameraMetadata(
        camera_name="front_camera",
        camera_id=CameraID.PCAM_F0,
        intrinsics=PinholeIntrinsics(fx=10.0, fy=10.0, cx=width / 2, cy=height / 2),
        distortion=PinholeDistortion(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0),
        width=width,
        height=height,
        camera_to_imu_se3=PoseSE3.identity(),
    )


def _make_segmentation_camera(
    label_map: np.ndarray,
    segmentation_label_class=WODPerceptionCameraSegmentationLabel,
    channel_type: CameraChannelType = CameraChannelType.SEMANTIC,
) -> Camera:
    height, width = label_map.shape
    metadata = SegmentationCameraMetadata(
        camera_metadata=_make_rgb_camera_metadata(width=width, height=height),
        segmentation_label_class=segmentation_label_class,
        channel_type=channel_type,
    )
    return Camera(
        metadata=metadata,
        image=label_map,
        camera_to_global_se3=PoseSE3.identity(),
        timestamp=Timestamp.from_us(0),
    )


# ----------------------------------------------------------------------------------------------------------------------
# colorize_semantic_label_map
# ----------------------------------------------------------------------------------------------------------------------


class TestColorizeSemanticLabelMap:
    def test_returns_rgb_uint8_with_same_hw(self):
        label_map = np.zeros((6, 8), dtype=np.uint8)
        rgb = colorize_semantic_label_map(label_map, WODPerceptionCameraSegmentationLabel)
        assert rgb.shape == (6, 8, 3)
        assert rgb.dtype == np.uint8

    def test_ids_map_to_cityscapes_colors(self):
        wod = WODPerceptionCameraSegmentationLabel
        label_map = np.array(
            [
                [wod.TYPE_CAR.value, wod.TYPE_PEDESTRIAN.value],
                [wod.TYPE_ROAD.value, wod.TYPE_UNDEFINED.value],
            ],
            dtype=np.uint8,
        )
        rgb = colorize_semantic_label_map(label_map, wod)
        assert tuple(rgb[0, 0]) == (0, 0, 142)  # car -> vehicle
        assert tuple(rgb[0, 1]) == (220, 20, 60)  # pedestrian -> person
        assert tuple(rgb[1, 0]) == (128, 64, 128)  # road
        assert tuple(rgb[1, 1]) == (0, 0, 0)  # undefined -> ignore (black)

    def test_color_matches_unified_default_palette(self):
        # The color of a raw id equals the palette color of its to_default() label.
        wod = WODPerceptionCameraSegmentationLabel
        label_map = np.array([[wod.TYPE_SIDEWALK.value]], dtype=np.uint8)
        rgb = colorize_semantic_label_map(label_map, wod)
        expected = DEFAULT_CAMERA_SEGMENTATION_RGB[wod.TYPE_SIDEWALK.to_default()]
        assert tuple(rgb[0, 0]) == expected

    def test_out_of_taxonomy_id_falls_back_to_other(self):
        # An id that is not a member of the label class is colored with the OTHER color.
        unknown_id = 200
        assert unknown_id not in {int(m) for m in WODPerceptionCameraSegmentationLabel}
        label_map = np.array([[unknown_id]], dtype=np.uint8)
        rgb = colorize_semantic_label_map(label_map, WODPerceptionCameraSegmentationLabel)
        assert tuple(rgb[0, 0]) == DEFAULT_CAMERA_SEGMENTATION_RGB[DefaultCameraSegmentationLabel.OTHER]

    def test_color_is_stable_per_id(self):
        wod = WODPerceptionCameraSegmentationLabel
        label_map = np.array([[wod.TYPE_CAR.value, wod.TYPE_ROAD.value, wod.TYPE_CAR.value]], dtype=np.uint8)
        rgb = colorize_semantic_label_map(label_map, wod)
        np.testing.assert_array_equal(rgb[0, 0], rgb[0, 2])  # both car
        assert not np.array_equal(rgb[0, 0], rgb[0, 1])  # car != road

    def test_handles_uint16_label_map(self):
        # KITTI-360 has 45 classes; a uint16 map of high-id classes must still color correctly.
        kitti = Kitti360CameraSegmentationLabel
        members = list(kitti)
        label_map = np.array([[int(members[-1]), int(members[0])]], dtype=np.uint16)
        rgb = colorize_semantic_label_map(label_map, kitti)
        assert rgb.shape == (1, 2, 3)
        assert tuple(rgb[0, 0]) == DEFAULT_CAMERA_SEGMENTATION_RGB[members[-1].to_default()]


# ----------------------------------------------------------------------------------------------------------------------
# Camera.rgb_image
# ----------------------------------------------------------------------------------------------------------------------


class TestCameraRgbImage:
    def test_semantic_camera_returns_colorized_image(self):
        wod = WODPerceptionCameraSegmentationLabel
        label_map = np.full((6, 8), wod.TYPE_VEGETATION.value, dtype=np.uint8)
        camera = _make_segmentation_camera(label_map, segmentation_label_class=wod)
        rgb = camera.rgb_image
        assert rgb.shape == (6, 8, 3)
        assert rgb.dtype == np.uint8
        assert tuple(rgb[0, 0]) == DEFAULT_CAMERA_SEGMENTATION_RGB[wod.TYPE_VEGETATION.to_default()]

    def test_semantic_matches_standalone_colorizer(self):
        wod = WODPerceptionCameraSegmentationLabel
        label_map = np.arange(48, dtype=np.uint8).reshape(6, 8) % len(list(wod))
        camera = _make_segmentation_camera(label_map, segmentation_label_class=wod)
        np.testing.assert_array_equal(camera.rgb_image, colorize_semantic_label_map(label_map, wod))

    def test_instance_camera_returns_colorized_image(self):
        label_map = np.array([[0, 1], [2, 1]], dtype=np.uint16)
        camera = _make_segmentation_camera(label_map, channel_type=CameraChannelType.INSTANCE)
        rgb = camera.rgb_image
        assert rgb.shape == (2, 2, 3)
        assert rgb.dtype == np.uint8
        assert tuple(rgb[0, 0]) == (0, 0, 0)  # id 0 (background) -> black
        assert tuple(rgb[0, 1]) == tuple(TABLEAU_20_RGB[1])  # same id -> same color
        np.testing.assert_array_equal(rgb[0, 1], rgb[1, 1])
        assert not np.array_equal(rgb[0, 1], rgb[1, 0])  # distinct ids -> distinct colors

    def test_instance_matches_standalone_colorizer(self):
        label_map = np.arange(48, dtype=np.uint16).reshape(6, 8)
        camera = _make_segmentation_camera(label_map, channel_type=CameraChannelType.INSTANCE)
        np.testing.assert_array_equal(camera.rgb_image, colorize_instance_label_map(label_map))


# ----------------------------------------------------------------------------------------------------------------------
# colorize_instance_label_map
# ----------------------------------------------------------------------------------------------------------------------


class TestColorizeInstanceLabelMap:
    def test_returns_rgb_uint8_with_same_hw(self):
        label_map = np.zeros((6, 8), dtype=np.uint16)
        rgb = colorize_instance_label_map(label_map)
        assert rgb.shape == (6, 8, 3)
        assert rgb.dtype == np.uint8

    def test_background_id_zero_is_black(self):
        rgb = colorize_instance_label_map(np.zeros((2, 2), dtype=np.uint16))
        assert np.all(rgb == 0)

    def test_color_is_stable_per_id_and_cycles(self):
        n = len(TABLEAU_20_RGB)
        # An id and the same id shifted by one palette period share a color (cycling).
        label_map = np.array([[1, 1 + n], [2, 3]], dtype=np.int64)
        rgb = colorize_instance_label_map(label_map)
        np.testing.assert_array_equal(rgb[0, 0], rgb[0, 1])
        assert tuple(rgb[0, 0]) == tuple(TABLEAU_20_RGB[1])
        assert not np.array_equal(rgb[1, 0], rgb[1, 1])  # ids 2 and 3 differ


# ----------------------------------------------------------------------------------------------------------------------
# Palette single-source-of-truth
# ----------------------------------------------------------------------------------------------------------------------


class TestPaletteDerivation:
    def test_visualization_colors_derive_from_datatypes_tuples(self):
        # The visualization Color dict must stay in lockstep with the data-layer RGB tuples it derives from.
        from py123d.visualization.color.default import DEFAULT_CAMERA_SEGMENTATION_COLORS

        assert set(DEFAULT_CAMERA_SEGMENTATION_COLORS) == set(DEFAULT_CAMERA_SEGMENTATION_RGB)
        for label, rgb in DEFAULT_CAMERA_SEGMENTATION_RGB.items():
            assert DEFAULT_CAMERA_SEGMENTATION_COLORS[label].rgb == rgb

    def test_every_default_label_has_a_color(self):
        for label in DefaultCameraSegmentationLabel:
            assert label in DEFAULT_CAMERA_SEGMENTATION_RGB
