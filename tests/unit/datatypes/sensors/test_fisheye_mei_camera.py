import unittest

import numpy as np

from py123d.datatypes.sensors.fisheye_mei_camera import (
    FisheyeMEICamera,
    FisheyeMEICameraMetadata,
    FisheyeMEICameraType,
    FisheyeMEIDistortion,
    FisheyeMEIDistortionIndex,
    FisheyeMEIProjection,
    FisheyeMEIProjectionIndex,
)
from py123d.geometry import PoseSE3


class TestFisheyeMEICameraType(unittest.TestCase):

    def test_camera_type_values(self):
        """Test that camera type enum has expected values."""
        self.assertEqual(FisheyeMEICameraType.FCAM_L.value, 0)
        self.assertEqual(FisheyeMEICameraType.FCAM_R.value, 1)

    def test_camera_type_from_int(self):
        """Test creating camera type from integer values."""
        self.assertEqual(FisheyeMEICameraType(0), FisheyeMEICameraType.FCAM_L)
        self.assertEqual(FisheyeMEICameraType(1), FisheyeMEICameraType.FCAM_R)

    def test_camera_type_members(self):
        """Test that all expected members exist."""
        members = list(FisheyeMEICameraType)
        self.assertEqual(len(members), 2)
        self.assertIn(FisheyeMEICameraType.FCAM_L, members)
        self.assertIn(FisheyeMEICameraType.FCAM_R, members)

    def test_camera_type_comparison(self):
        """Test comparison between camera types."""
        self.assertNotEqual(FisheyeMEICameraType.FCAM_L, FisheyeMEICameraType.FCAM_R)
        self.assertEqual(FisheyeMEICameraType.FCAM_L, FisheyeMEICameraType.FCAM_L)


class TestFisheyeMEIDistortion(unittest.TestCase):

    def test_distortion_initialization(self):
        """Test distortion parameter initialization."""
        distortion = FisheyeMEIDistortion(k1=0.1, k2=0.2, p1=0.3, p2=0.4)
        self.assertEqual(distortion.k1, 0.1)
        self.assertEqual(distortion.k2, 0.2)
        self.assertEqual(distortion.p1, 0.3)
        self.assertEqual(distortion.p2, 0.4)

    def test_distortion_from_array(self):
        """Test creating distortion from array."""
        array = np.array([0.1, 0.2, 0.3, 0.4])
        distortion = FisheyeMEIDistortion.from_array(array)
        self.assertEqual(distortion.k1, 0.1)
        self.assertEqual(distortion.k2, 0.2)
        self.assertEqual(distortion.p1, 0.3)
        self.assertEqual(distortion.p2, 0.4)

    def test_distortion_from_array_copy(self):
        """Test that from_array copies data by default."""
        array = np.array([0.1, 0.2, 0.3, 0.4])
        distortion = FisheyeMEIDistortion.from_array(array, copy=True)
        array[0] = 999.0
        self.assertEqual(distortion.k1, 0.1)

    def test_distortion_from_array_no_copy(self):
        """Test that from_array can avoid copying."""
        array = np.array([0.1, 0.2, 0.3, 0.4])
        distortion = FisheyeMEIDistortion.from_array(array, copy=False)
        array[0] = 999.0
        self.assertEqual(distortion.k1, 999.0)

    def test_distortion_array_property(self):
        """Test array property returns correct values."""
        distortion = FisheyeMEIDistortion(k1=0.1, k2=0.2, p1=0.3, p2=0.4)
        array = distortion.array
        self.assertEqual(len(array), 4)
        np.testing.assert_array_equal(array, [0.1, 0.2, 0.3, 0.4])

    def test_distortion_index_mapping(self):
        """Test that distortion indices map correctly."""
        distortion = FisheyeMEIDistortion(k1=0.1, k2=0.2, p1=0.3, p2=0.4)
        self.assertEqual(distortion.array[FisheyeMEIDistortionIndex.K1], 0.1)
        self.assertEqual(distortion.array[FisheyeMEIDistortionIndex.K2], 0.2)
        self.assertEqual(distortion.array[FisheyeMEIDistortionIndex.P1], 0.3)
        self.assertEqual(distortion.array[FisheyeMEIDistortionIndex.P2], 0.4)


class TestFisheyeMEIProjection(unittest.TestCase):

    def test_projection_initialization(self):
        """Test projection parameter initialization."""
        projection = FisheyeMEIProjection(gamma1=1.0, gamma2=2.0, u0=3.0, v0=4.0)
        self.assertEqual(projection.gamma1, 1.0)
        self.assertEqual(projection.gamma2, 2.0)
        self.assertEqual(projection.u0, 3.0)
        self.assertEqual(projection.v0, 4.0)

    def test_projection_from_array(self):
        """Test creating projection from array."""
        array = np.array([1.0, 2.0, 3.0, 4.0])
        projection = FisheyeMEIProjection.from_array(array)
        self.assertEqual(projection.gamma1, 1.0)
        self.assertEqual(projection.gamma2, 2.0)
        self.assertEqual(projection.u0, 3.0)
        self.assertEqual(projection.v0, 4.0)

    def test_projection_from_array_copy(self):
        """Test that from_array copies data by default."""
        array = np.array([1.0, 2.0, 3.0, 4.0])
        projection = FisheyeMEIProjection.from_array(array, copy=True)
        array[0] = 999.0
        self.assertEqual(projection.gamma1, 1.0)

    def test_projection_from_array_no_copy(self):
        """Test that from_array can avoid copying."""
        array = np.array([1.0, 2.0, 3.0, 4.0])
        projection = FisheyeMEIProjection.from_array(array, copy=False)
        array[0] = 999.0
        self.assertEqual(projection.gamma1, 999.0)

    def test_projection_array_property(self):
        """Test array property returns correct values."""
        projection = FisheyeMEIProjection(gamma1=1.0, gamma2=2.0, u0=3.0, v0=4.0)
        array = projection.array
        self.assertEqual(len(array), 4)
        np.testing.assert_array_equal(array, [1.0, 2.0, 3.0, 4.0])

    def test_projection_index_mapping(self):
        """Test that projection indices map correctly."""
        projection = FisheyeMEIProjection(gamma1=1.0, gamma2=2.0, u0=3.0, v0=4.0)
        self.assertEqual(projection.array[FisheyeMEIProjectionIndex.GAMMA1], 1.0)
        self.assertEqual(projection.array[FisheyeMEIProjectionIndex.GAMMA2], 2.0)
        self.assertEqual(projection.array[FisheyeMEIProjectionIndex.U0], 3.0)
        self.assertEqual(projection.array[FisheyeMEIProjectionIndex.V0], 4.0)


class TestFisheyeMEICameraMetadata(unittest.TestCase):

    def test_metadata_initialization(self):
        """Test metadata initialization with all parameters."""
        distortion = FisheyeMEIDistortion(k1=0.1, k2=0.2, p1=0.3, p2=0.4)
        projection = FisheyeMEIProjection(gamma1=1.0, gamma2=2.0, u0=3.0, v0=4.0)
        metadata = FisheyeMEICameraMetadata(
            camera_type=FisheyeMEICameraType.FCAM_L,
            mirror_parameter=0.5,
            distortion=distortion,
            projection=projection,
            width=1920,
            height=1080,
        )
        self.assertEqual(metadata.camera_type, FisheyeMEICameraType.FCAM_L)
        self.assertEqual(metadata.mirror_parameter, 0.5)
        self.assertEqual(metadata.distortion, distortion)
        self.assertEqual(metadata.projection, projection)
        self.assertEqual(metadata.aspect_ratio, 1920 / 1080)

    def test_metadata_initialization_with_none(self):
        """Test metadata initialization with None distortion and projection."""
        metadata = FisheyeMEICameraMetadata(
            camera_type=FisheyeMEICameraType.FCAM_R,
            mirror_parameter=None,
            distortion=None,
            projection=None,
            width=640,
            height=480,
        )
        self.assertEqual(metadata.camera_type, FisheyeMEICameraType.FCAM_R)
        self.assertIsNone(metadata.mirror_parameter)
        self.assertIsNone(metadata.distortion)
        self.assertIsNone(metadata.projection)
        self.assertEqual(metadata.aspect_ratio, 640 / 480)

    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        distortion = FisheyeMEIDistortion(k1=0.1, k2=0.2, p1=0.3, p2=0.4)
        projection = FisheyeMEIProjection(gamma1=1.0, gamma2=2.0, u0=3.0, v0=4.0)
        metadata = FisheyeMEICameraMetadata(
            camera_type=FisheyeMEICameraType.FCAM_L,
            mirror_parameter=0.5,
            distortion=distortion,
            projection=projection,
            width=1920,
            height=1080,
        )
        result = metadata.to_dict()
        self.assertEqual(result["camera_type"], 0)
        self.assertEqual(result["mirror_parameter"], 0.5)
        self.assertEqual(result["distortion"], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(result["projection"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["width"], 1920)
        self.assertEqual(result["height"], 1080)

    def test_metadata_to_dict_with_none(self):
        """Test converting metadata with None values to dictionary."""
        metadata = FisheyeMEICameraMetadata(
            camera_type=FisheyeMEICameraType.FCAM_R,
            mirror_parameter=None,
            distortion=None,
            projection=None,
            width=640,
            height=480,
        )
        result = metadata.to_dict()
        self.assertEqual(result["camera_type"], 1)
        self.assertIsNone(result["mirror_parameter"])
        self.assertIsNone(result["distortion"])
        self.assertIsNone(result["projection"])
        self.assertEqual(result["width"], 640)
        self.assertEqual(result["height"], 480)

    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        data = {
            "camera_type": 0,
            "mirror_parameter": 0.5,
            "distortion": [0.1, 0.2, 0.3, 0.4],
            "projection": [1.0, 2.0, 3.0, 4.0],
            "width": 1920,
            "height": 1080,
        }
        metadata = FisheyeMEICameraMetadata.from_dict(data)
        self.assertEqual(metadata.camera_type, FisheyeMEICameraType.FCAM_L)
        self.assertEqual(metadata.mirror_parameter, 0.5)
        self.assertEqual(metadata.distortion.k1, 0.1)
        self.assertEqual(metadata.projection.gamma1, 1.0)
        self.assertEqual(metadata.aspect_ratio, 1920 / 1080)

    def test_metadata_from_dict_with_none(self):
        """Test creating metadata from dictionary with None values."""
        data = {
            "camera_type": 1,
            "mirror_parameter": None,
            "distortion": None,
            "projection": None,
            "width": 640,
            "height": 480,
        }
        metadata = FisheyeMEICameraMetadata.from_dict(data)
        self.assertEqual(metadata.camera_type, FisheyeMEICameraType.FCAM_R)
        self.assertIsNone(metadata.mirror_parameter)
        self.assertIsNone(metadata.distortion)
        self.assertIsNone(metadata.projection)

    def test_metadata_roundtrip(self):
        """Test that to_dict and from_dict are inverses."""
        distortion = FisheyeMEIDistortion(k1=0.1, k2=0.2, p1=0.3, p2=0.4)
        projection = FisheyeMEIProjection(gamma1=1.0, gamma2=2.0, u0=3.0, v0=4.0)
        metadata = FisheyeMEICameraMetadata(
            camera_type=FisheyeMEICameraType.FCAM_L,
            mirror_parameter=0.5,
            distortion=distortion,
            projection=projection,
            width=1920,
            height=1080,
        )
        data_dict = metadata.to_dict()
        metadata_restored = FisheyeMEICameraMetadata.from_dict(data_dict)
        self.assertEqual(metadata.camera_type, metadata_restored.camera_type)
        self.assertEqual(metadata.mirror_parameter, metadata_restored.mirror_parameter)
        np.testing.assert_array_equal(metadata.distortion.array, metadata_restored.distortion.array)
        np.testing.assert_array_equal(metadata.projection.array, metadata_restored.projection.array)
        self.assertEqual(metadata.aspect_ratio, metadata_restored.aspect_ratio)

    def test_aspect_ratio_calculation(self):
        """Test aspect ratio calculation."""
        metadata = FisheyeMEICameraMetadata(
            camera_type=FisheyeMEICameraType.FCAM_L,
            mirror_parameter=0.5,
            distortion=None,
            projection=None,
            width=1920,
            height=1080,
        )
        self.assertAlmostEqual(metadata.aspect_ratio, 16 / 9, places=5)


class TestFisheyeMEICamera(unittest.TestCase):

    def test_camera_initialization(self):
        """Test FisheyeMEICamera initialization."""

        metadata = FisheyeMEICameraMetadata(
            camera_type=FisheyeMEICameraType.FCAM_L,
            mirror_parameter=0.5,
            distortion=FisheyeMEIDistortion(k1=0.1, k2=0.2, p1=0.3, p2=0.4),
            projection=FisheyeMEIProjection(gamma1=1.0, gamma2=2.0, u0=3.0, v0=4.0),
            width=1920,
            height=1080,
        )
        image = np.zeros((1080, 1920), dtype=np.uint8)
        extrinsic = PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)

        camera = FisheyeMEICamera(metadata=metadata, image=image, extrinsic=extrinsic)

        self.assertEqual(camera.metadata, metadata)
        np.testing.assert_array_equal(camera.image, image)
        self.assertEqual(camera.extrinsic, extrinsic)

    def test_camera_metadata_property(self):
        """Test that metadata property returns correct metadata."""

        metadata = FisheyeMEICameraMetadata(
            camera_type=FisheyeMEICameraType.FCAM_R,
            mirror_parameter=0.8,
            distortion=None,
            projection=None,
            width=640,
            height=480,
        )
        image = np.ones((480, 640), dtype=np.uint8)
        extrinsic = PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)

        camera = FisheyeMEICamera(metadata=metadata, image=image, extrinsic=extrinsic)

        self.assertIs(camera.metadata, metadata)
        self.assertEqual(camera.metadata.camera_type, FisheyeMEICameraType.FCAM_R)

    def test_camera_image_property(self):
        """Test that image property returns correct image."""

        metadata = FisheyeMEICameraMetadata(
            camera_type=FisheyeMEICameraType.FCAM_L,
            mirror_parameter=0.5,
            distortion=None,
            projection=None,
            width=640,
            height=480,
        )
        image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        extrinsic = PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)

        camera = FisheyeMEICamera(metadata=metadata, image=image, extrinsic=extrinsic)

        np.testing.assert_array_equal(camera.image, image)
        self.assertEqual(camera.image.dtype, np.uint8)

    def test_camera_extrinsic_property(self):
        """Test that extrinsic property returns correct pose."""

        metadata = FisheyeMEICameraMetadata(
            camera_type=FisheyeMEICameraType.FCAM_L,
            mirror_parameter=0.5,
            distortion=None,
            projection=None,
            width=640,
            height=480,
        )
        image = np.zeros((480, 640), dtype=np.uint8)
        extrinsic = PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)

        camera = FisheyeMEICamera(metadata=metadata, image=image, extrinsic=extrinsic)

        self.assertIs(camera.extrinsic, extrinsic)

    def test_camera_with_color_image(self):
        """Test camera with color (3-channel) image."""

        metadata = FisheyeMEICameraMetadata(
            camera_type=FisheyeMEICameraType.FCAM_L,
            mirror_parameter=0.5,
            distortion=None,
            projection=None,
            width=640,
            height=480,
        )
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        extrinsic = PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)

        camera = FisheyeMEICamera(metadata=metadata, image=image, extrinsic=extrinsic)

        self.assertEqual(camera.image.shape, (480, 640, 3))
