import unittest

from py123d.datatypes.vehicle_state.vehicle_parameters import VehicleParameters


class TestVehicleParameters(unittest.TestCase):

    def setUp(self):
        self.params = VehicleParameters(
            vehicle_name="test_vehicle",
            width=2.0,
            length=5.0,
            height=1.8,
            wheel_base=3.0,
            rear_axle_to_center_vertical=0.5,
            rear_axle_to_center_longitudinal=1.5,
        )

    def test_initialization(self):
        self.assertEqual(self.params.vehicle_name, "test_vehicle")
        self.assertEqual(self.params.width, 2.0)
        self.assertEqual(self.params.length, 5.0)
        self.assertEqual(self.params.height, 1.8)
        self.assertEqual(self.params.wheel_base, 3.0)
        self.assertEqual(self.params.rear_axle_to_center_vertical, 0.5)
        self.assertEqual(self.params.rear_axle_to_center_longitudinal, 1.5)

    def test_half_width(self):
        self.assertEqual(self.params.half_width, 1.0)

    def test_half_length(self):
        self.assertEqual(self.params.half_length, 2.5)

    def test_half_height(self):
        self.assertEqual(self.params.half_height, 0.9)

    def test_to_dict(self):
        result = self.params.to_dict()
        expected = {
            "vehicle_name": "test_vehicle",
            "width": 2.0,
            "length": 5.0,
            "height": 1.8,
            "wheel_base": 3.0,
            "rear_axle_to_center_vertical": 0.5,
            "rear_axle_to_center_longitudinal": 1.5,
        }
        self.assertEqual(result, expected)

    def test_from_dict(self):
        data = {
            "vehicle_name": "from_dict_vehicle",
            "width": 1.5,
            "length": 4.0,
            "height": 1.6,
            "wheel_base": 2.5,
            "rear_axle_to_center_vertical": 0.4,
            "rear_axle_to_center_longitudinal": 1.2,
        }
        params = VehicleParameters.from_dict(data)
        self.assertEqual(params.vehicle_name, "from_dict_vehicle")
        self.assertEqual(params.width, 1.5)
        self.assertEqual(params.length, 4.0)
        self.assertEqual(params.height, 1.6)
        self.assertEqual(params.wheel_base, 2.5)
        self.assertEqual(params.rear_axle_to_center_vertical, 0.4)
        self.assertEqual(params.rear_axle_to_center_longitudinal, 1.2)

    def test_from_dict_to_dict_round_trip(self):
        original_dict = self.params.to_dict()
        recreated_params = VehicleParameters.from_dict(original_dict)
        recreated_dict = recreated_params.to_dict()
        self.assertEqual(original_dict, recreated_dict)
