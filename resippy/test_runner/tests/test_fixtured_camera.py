from __future__ import division

import unittest
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.fixtured_camera import FixturedCamera
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.pinhole_camera import PinholeCamera
import resippy.utils.photogrammetry_utils as photogram_utils
import numpy as np

world_x = 0
world_y = 0
world_z = 100


class TestFixturedCamera(unittest.TestCase):
    
    def test_fixture_roll_boresight(self):
        fixtured_cam = FixturedCamera()
        fixtured_cam.set_relative_camera_xyz(0, 0, 0)
        fixtured_cam.set_fixture_orientation_by_roll_pitch_yaw(5, 0, 0, roll_units='degrees')
        fixtured_cam.set_boresight_matrix_from_camera_relative_rpy_params(-5, 0, 0, roll_units='degrees')
        camera_M = fixtured_cam.get_camera_absolute_M_matrix()
        roll, pitch, yaw = photogram_utils.solve_for_omega_phi_kappa(camera_M)
        assert np.isclose(roll, 0)
        assert np.isclose(pitch, 0)
        assert np.isclose(yaw, 0)
        print("setting fixture roll to 5 degrees and camera boresight roll to -5 degrees results in a zero camera roation")
        print("roll boresight test passed.")

    def test_fixture_pitch_boresight(self):
        fixtured_cam = FixturedCamera()
        fixtured_cam.set_relative_camera_xyz(0, 0, 0)
        fixtured_cam.set_fixture_orientation_by_roll_pitch_yaw(0, 5, 0, roll_units='degrees')
        fixtured_cam.set_boresight_matrix_from_camera_relative_rpy_params(0, -5, 0, roll_units='degrees')
        camera_M = fixtured_cam.get_camera_absolute_M_matrix()
        roll, pitch, yaw = photogram_utils.solve_for_omega_phi_kappa(camera_M)
        assert np.isclose(roll, 0)
        assert np.isclose(pitch, 0)
        assert np.isclose(yaw, 0)
        print("setting fixture pitch to 5 degrees and camera boresight pitch to -5 degrees results in a zero camera roation")
        print("pitch boresight test passed.")

    def test_fixture_yaw_boresight(self):
        fixtured_cam_1 = FixturedCamera()
        fixtured_cam_1.set_relative_camera_xyz(0, 0, 100)
        fixtured_cam_1.set_fixture_orientation_by_roll_pitch_yaw(0, 0, 0, roll_units='degrees')
        fixtured_cam_1.set_boresight_matrix_from_camera_relative_rpy_params(0, 0, 0, roll_units='degrees')

        cam1_x, cam1_y, cam1_z = fixtured_cam_1.get_camera_absolute_xyz()
        cam1_M = fixtured_cam_1.get_camera_absolute_M_matrix()

        omega1, phi1, kappa1 = photogram_utils.solve_for_omega_phi_kappa(cam1_M)

        pinhole_cam1 = PinholeCamera()
        pinhole_cam1.init_pinhole_from_coeffs(cam1_x, cam1_y, cam1_z, omega1, phi1, kappa1,
                                              focal_length=50, focal_length_units='millimeters')

        fpa_x1, fpa_y1 = pinhole_cam1.world_to_image_plane(50, 0, 0)

        fixtured_cam_1.set_fixture_orientation_by_roll_pitch_yaw(0, 0, 5, yaw_units='degrees')

        cam2_x, cam2_y, cam2_z = fixtured_cam_1.get_camera_absolute_xyz()
        cam2_M = fixtured_cam_1.get_camera_absolute_M_matrix()
        omega2, phi2, kappa2 = photogram_utils.solve_for_omega_phi_kappa(cam2_M)

        pinhole_cam2 = PinholeCamera()
        pinhole_cam2.init_pinhole_from_coeffs(cam2_x, cam2_y, cam2_z, omega2, phi2, kappa2,
                                              focal_length=50, focal_length_units='millimeters')
        fpa_x2, fpa_y2 = pinhole_cam2.world_to_image_plane(50, 0, 0)

        fixtured_cam_1.set_boresight_matrix_from_camera_relative_rpy_params(0, 0, -5, yaw_units='degrees')

        cam3_x, cam3_y, cam3_z = fixtured_cam_1.get_camera_absolute_xyz()
        cam3_M = fixtured_cam_1.get_camera_absolute_M_matrix()
        omega3, phi3, kappa3 = photogram_utils.solve_for_omega_phi_kappa(cam3_M)

        pinhole_cam2 = PinholeCamera()
        pinhole_cam2.init_pinhole_from_coeffs(cam3_x, cam3_y, cam3_z, omega3, phi3, kappa3,
                                              focal_length=50, focal_length_units='millimeters')
        fpa_x3, fpa_y3 = pinhole_cam2.world_to_image_plane(50, 0, 0)

        assert np.isclose(fpa_y1, fpa_y3)
        assert np.isclose(fpa_x1, fpa_x3)
        assert not np.isclose(fpa_x1, fpa_x2)
        assert not np.isclose(fpa_y1, fpa_y2)
        print("setting fixture to 5 degrees and camera boresight yaw to -5 degrees results in a zero camera rotation")
        print("yaw boresight test passed.")


if __name__ == '__main__':
    unittest.main()
