from __future__ import division

import unittest
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.fixtured_camera import FixturedCamera
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.pinhole_camera import PinholeCamera

world_x = 0
world_y = 0
world_z = 100


class TestFixturedCamera(unittest.TestCase):

    def test_relatitive_roll_with_ylever_arm(self):
        pinhole_cam = PinholeCamera()
        fixtured_camera_0 = FixturedCamera()
        fixtured_camera_0.set_fixture_by_roll_pitch_yaw(0, 0, 0)

        fixtured_camera_0.set_boresight_matrix_from_camera_relative_rpy_params(0, 0, 0, roll_units='degrees')
        fixtured_camera_0.set_relative_camera_xyz(0, 1, 0)
        fixtured_camera_0.get_camera_absolute_xyz()

        fixtured_camera_1 = FixturedCamera()
        fixtured_camera_1.set_fixture_by_roll_pitch_yaw(0, 0, 0)

        fixtured_camera_1.set_boresight_matrix_from_camera_relative_rpy_params(1, 0, 0, roll_units='degrees')
        fixtured_camera_1.set_relative_camera_xyz(0, 1, 0)
        fixtured_camera_1.get_camera_absolute_xyz()

        pinhole_cam_0 = fixtured_camera_0.get_camera_absolute_xyz()
        pinhole_cam_1 = fixtured_camera_1.get_camera_absolute_xyz()

        # TODO the Z coordinate should be higher or lower for cam0 than cam1, need to figure out which one is right
        stop = 1


if __name__ == '__main__':
    unittest.main()
