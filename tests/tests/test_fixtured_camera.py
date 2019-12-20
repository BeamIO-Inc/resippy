from __future__ import division

import unittest
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.fixtured_camera import FixturedCamera
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.pinhole_camera import PinholeCamera
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.ideal_pinhole_fpa_local_utm_point_calc import IdealPinholeFpaLocalUtmPointCalc
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
        # TODO: use higher precision floats if we want better absolute accuracy here
        assert np.isclose(roll, 0, atol=1.e-7)
        assert np.isclose(pitch, 0, atol=1.e-7)
        assert np.isclose(yaw, 0, atol=1.e-7)
        print("setting fixture pitch to 5 degrees and camera boresight pitch to -5 degrees results in a zero camera roation")
        print("pitch boresight test passed.")

    def test_fixture_yaw_boresight(self):
        fixtured_cam = FixturedCamera()
        fixtured_cam.set_relative_camera_xyz(0, 0, 0)
        fixtured_cam.set_fixture_orientation_by_roll_pitch_yaw(0, 0, 5, yaw_units='degrees')
        fixtured_cam.set_boresight_matrix_from_camera_relative_rpy_params(0, 0, -5, yaw_units='degrees')
        camera_M = fixtured_cam.get_camera_absolute_M_matrix()
        roll, pitch, yaw = photogram_utils.solve_for_omega_phi_kappa(camera_M)
        assert np.isclose(roll, 0)
        assert np.isclose(pitch, 0)
        assert np.isclose(yaw, 0)
        print("setting fixture yaw to 5 degrees and camera boresight yaw to -5 degrees results in a zero camera roation")
        print("pitch boresight test passed.")

    def test_fixture_yaw_boresight2(self):
        fixtured_cam_1 = FixturedCamera()
        fixtured_cam_1.set_relative_camera_xyz(0, 0, 0)
        fixtured_cam_1.set_fixture_orientation_by_roll_pitch_yaw(0, 0, 0, roll_units='degrees')
        fixtured_cam_1.set_boresight_matrix_from_camera_relative_rpy_params(0, 0, 0, roll_units='degrees')
        fixtured_cam_1.set_fixture_xyz(0, 0, 100)

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

    def test_y_lever_arm_offset(self):
        lat = 43.085898
        lon = -77.677624
        sensor_alt = 100

        temp_point_calc = IdealPinholeFpaLocalUtmPointCalc.init_from_wgs84_params(lon, lat, sensor_alt,
                                                                                 0, 0, 0,
                                                                                 1000, 1000,
                                                                                 5, 5, 1)

        fixture_local_lon, fixture_local_lat = temp_point_calc.get_approximate_lon_lat_center()

        fixtured_cam = FixturedCamera()
        fixtured_cam.set_fixture_xyz(fixture_local_lon, fixture_local_lat, sensor_alt)
        fixtured_cam.set_relative_camera_xyz(0, 1, 0)
        fixtured_cam.set_fixture_orientation_by_roll_pitch_yaw(0, 0, 0,
                                                               roll_units='degrees',
                                                               pitch_units='degrees',
                                                               yaw_units='degrees')
        fixtured_cam.set_boresight_matrix_from_camera_relative_rpy_params(0, 0, 0)
        camera_M = fixtured_cam.get_camera_absolute_M_matrix()
        camera_x, camera_y, camera_z = fixtured_cam.get_camera_absolute_xyz()

        omega, phi, kappa = photogram_utils.solve_for_omega_phi_kappa(camera_M)
        pinhole_cam = PinholeCamera()
        pinhole_cam.init_pinhole_from_coeffs(camera_x, camera_y, camera_z, omega, phi, kappa,
                                             1, focal_length_units='meters')

        fpa_point_calc = IdealPinholeFpaLocalUtmPointCalc.init_from_local_params(camera_x, camera_y, camera_z,
                                                                                 temp_point_calc.get_projection(),
                                                                                 omega, phi, kappa,
                                                                                 1000, 1000,
                                                                                 5, 5, 1)

        fpa_center_lon, fpa_center_lat = fpa_point_calc.pixel_x_y_alt_to_lon_lat(500, 500, 0)

        fixture_x, fixture_y, fixture_z = fixtured_cam.get_fixture_xyz()

        assert np.isclose(fixture_x, fpa_center_lon)
        assert np.isclose(fixture_y, fpa_center_lat)
        assert np.isclose(camera_x, fixture_x)
        assert np.isclose(camera_y - fixture_y, 1)
        assert np.isclose(camera_z, fixture_z)

        print("y lever arm is correctly returned by the fixtured camera")

    def test_y_lever_arm_offset(self):
        lat = 43.085898
        lon = -77.677624
        sensor_alt = 100

        temp_point_calc = IdealPinholeFpaLocalUtmPointCalc.init_from_wgs84_params(lon, lat, sensor_alt,
                                                                                 0, 0, 0,
                                                                                 1000, 1000,
                                                                                 5, 5, 1)

        fixture_local_lon, fixture_local_lat = temp_point_calc.get_approximate_lon_lat_center()

        fixtured_cam = FixturedCamera()
        fixtured_cam.set_fixture_xyz(fixture_local_lon, fixture_local_lat, sensor_alt)
        fixtured_cam.set_relative_camera_xyz(0, 1, 0)
        fixtured_cam.set_fixture_orientation_by_roll_pitch_yaw(10, 0, 0,
                                                               roll_units='degrees',
                                                               pitch_units='degrees',
                                                               yaw_units='degrees')
        fixtured_cam.set_boresight_matrix_from_camera_relative_rpy_params(0, 0, 0)
        camera_M = fixtured_cam.get_camera_absolute_M_matrix()
        camera_x, camera_y, camera_z = fixtured_cam.get_camera_absolute_xyz()

        omega, phi, kappa = photogram_utils.solve_for_omega_phi_kappa(camera_M)
        pinhole_cam = PinholeCamera()
        pinhole_cam.init_pinhole_from_coeffs(camera_x, camera_y, camera_z, omega, phi, kappa,
                                             1, focal_length_units='meters')

        fpa_point_calc = IdealPinholeFpaLocalUtmPointCalc.init_from_local_params(camera_x, camera_y, camera_z,
                                                                                 temp_point_calc.get_projection(),
                                                                                 omega, phi, kappa,
                                                                                 1000, 1000,
                                                                                 5, 5, 1)

        fpa_center_lon, fpa_center_lat = fpa_point_calc.pixel_x_y_alt_to_lon_lat(500, 500, 0)

        fixture_x, fixture_y, fixture_z = fixtured_cam.get_fixture_xyz()

        assert np.isclose(fixture_x, camera_x)
        assert camera_z > fixture_z
        assert camera_y < (fixture_y+1)
        assert np.isclose(np.square(camera_y - fixture_y) + np.square(camera_z - fixture_z), 1.0)
        assert np.isclose(fpa_center_lon, camera_x)
        assert fpa_center_lat > camera_y

        print("positive roll results in camera center pointing to a higher latitude")
        print("makes sense by right-hand-rule conventions.")
        print("a positive y lever arm offset with a positive roll results in the camera moving higher in altitude and lower in latitude.")
        print("test passed.")

    def test_x_lever_arm_offset(self):
        lat = 43.085898
        lon = -77.677624
        sensor_alt = 100

        temp_point_calc = IdealPinholeFpaLocalUtmPointCalc.init_from_wgs84_params(lon, lat, sensor_alt,
                                                                                 0, 0, 0,
                                                                                 1000, 1000,
                                                                                 5, 5, 1)

        fixture_local_lon, fixture_local_lat = temp_point_calc.get_approximate_lon_lat_center()

        fixtured_cam = FixturedCamera()
        fixtured_cam.set_fixture_xyz(fixture_local_lon, fixture_local_lat, sensor_alt)
        fixtured_cam.set_relative_camera_xyz(1, 0, 0)
        fixtured_cam.set_fixture_orientation_by_roll_pitch_yaw(0, 10, 0,
                                                               roll_units='degrees',
                                                               pitch_units='degrees',
                                                               yaw_units='degrees')
        fixtured_cam.set_boresight_matrix_from_camera_relative_rpy_params(0, 0, 0)
        camera_M = fixtured_cam.get_camera_absolute_M_matrix()
        camera_x, camera_y, camera_z = fixtured_cam.get_camera_absolute_xyz()

        omega, phi, kappa = photogram_utils.solve_for_omega_phi_kappa(camera_M)
        pinhole_cam = PinholeCamera()
        pinhole_cam.init_pinhole_from_coeffs(camera_x, camera_y, camera_z, omega, phi, kappa,
                                             1, focal_length_units='meters')

        fpa_point_calc = IdealPinholeFpaLocalUtmPointCalc.init_from_local_params(camera_x, camera_y, camera_z,
                                                                                 temp_point_calc.get_projection(),
                                                                                 omega, phi, kappa,
                                                                                 1000, 1000,
                                                                                 5, 5, 1)

        fpa_center_lon, fpa_center_lat = fpa_point_calc.pixel_x_y_alt_to_lon_lat(500, 500, 0)

        fixture_x, fixture_y, fixture_z = fixtured_cam.get_fixture_xyz()

        assert np.isclose(fixture_y, camera_y)
        assert camera_z < fixture_z
        assert camera_x < (fixture_x+1)
        assert np.isclose(np.square(camera_x - fixture_x) + np.square(camera_z - fixture_z), 1.0)
        assert np.isclose(fpa_center_lat, camera_y)
        assert fpa_center_lon < camera_x

        print("positive pitch results in camera center pointing to a higher latitude")
        print("makes sense by right-hand-rule conventions.")
        print("a positive y lever arm offset with a positive roll results in the camera moving higher in altitude and lower in latitude.")
        print("test passed.")

    def test_x_lever_arm_offset_yaw(self):
        lat = 43.085898
        lon = -77.677624
        sensor_alt = 100

        temp_point_calc = IdealPinholeFpaLocalUtmPointCalc.init_from_wgs84_params(lon, lat, sensor_alt,
                                                                                 0, 0, 0,
                                                                                 1000, 1000,
                                                                                 5, 5, 1)

        fixture_local_lon, fixture_local_lat = temp_point_calc.get_approximate_lon_lat_center()

        fixtured_cam = FixturedCamera()
        fixtured_cam.set_fixture_xyz(fixture_local_lon, fixture_local_lat, sensor_alt)
        fixtured_cam.set_relative_camera_xyz(1, 0, 0)
        fixtured_cam.set_fixture_orientation_by_roll_pitch_yaw(0, 0, 10,
                                                               roll_units='degrees',
                                                               pitch_units='degrees',
                                                               yaw_units='degrees')
        fixtured_cam.set_boresight_matrix_from_camera_relative_rpy_params(0, 0, 0)
        camera_M = fixtured_cam.get_camera_absolute_M_matrix()
        camera_x, camera_y, camera_z = fixtured_cam.get_camera_absolute_xyz()

        omega, phi, kappa = photogram_utils.solve_for_omega_phi_kappa(camera_M)
        pinhole_cam = PinholeCamera()
        pinhole_cam.init_pinhole_from_coeffs(camera_x, camera_y, camera_z, omega, phi, kappa,
                                             1, focal_length_units='meters')

        fpa_point_calc1 = IdealPinholeFpaLocalUtmPointCalc.init_from_local_params(camera_x, camera_y, camera_z,
                                                                                 temp_point_calc.get_projection(),
                                                                                 omega, phi, 0,
                                                                                 1000, 1000,
                                                                                 5, 5, 1)

        fpa_point_calc2 = IdealPinholeFpaLocalUtmPointCalc.init_from_local_params(camera_x, camera_y, camera_z,
                                                                                 temp_point_calc.get_projection(),
                                                                                 omega, phi, kappa,
                                                                                 1000, 1000,
                                                                                 5, 5, 1)

        fpa_lon_1, fpa_lat_1 = fpa_point_calc1.pixel_x_y_alt_to_lon_lat(1000, 500, 0)
        fpa_lon_2, fpa_lat_2 = fpa_point_calc2.pixel_x_y_alt_to_lon_lat(1000, 500, 0)

        fixture_x, fixture_y, fixture_z = fixtured_cam.get_fixture_xyz()

        assert np.isclose(fpa_lat_1, camera_y)
        assert fpa_lon_1 > camera_x
        print("last pixel in the array is in the positive longitude direction at roll=0, pitch=0, yaw=0")
        assert fpa_lon_2 < fpa_lon_1
        assert fpa_lat_2 > fpa_lat_1
        print("a positive yaw rotation results seem correct from the perspective of pixel shift")
        assert camera_x < (fixture_x + 1)
        assert camera_y > fixture_y
        print("a positive yaw moves a camera originally located at (1, 0, 0) relative to the fixture to a greater y location and lower x location.")
        print("yaw rotation tests passed.")


if __name__ == '__main__':
    unittest.main()
