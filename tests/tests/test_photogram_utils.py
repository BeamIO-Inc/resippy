from __future__ import division

import unittest
import resippy.utils.photogrammetry_utils as photogram
import resippy.photogrammetry.crs_defs as crs_defs
from shapely.wkt import loads as wkt_loader
import numpy as np
from pyproj import transform as proj_transform


class TestCrsTools(unittest.TestCase):

    def test_reproject_geometry(self):
        world_poly_4326 = wkt_loader("POLYGON((-77.12352759307811 38.969453670463515,-76.92302710479686 "
                                     "38.969453670463515,-76.92302710479686 38.834794725571776,-77.12352759307811 "
                                     "38.834794725571776,-77.12352759307811 38.969453670463515))")
        world_poly_3857 = photogram.reproject_geometry(world_poly_4326, crs_defs.PROJ_4326, crs_defs.PROJ_3857)
        new_poly_4326 = photogram.reproject_geometry(world_poly_3857, crs_defs.PROJ_3857, crs_defs.PROJ_4326)
        assert not world_poly_4326.almost_equals(world_poly_3857)
        assert world_poly_4326.almost_equals(new_poly_4326)
        print("reproject geometry test passed")

    # TODO come up with a better test for this, maybe based on calculated distances in a local coordinate system
    def test_shift_points_by_meters_test(self):
        east_shift_meters = 1
        north_shift_meters = 1
        world_poly_4326 = wkt_loader("POLYGON((-77.12352759307811 38.969453670463515,-76.92302710479686 "
                                     "38.969453670463515,-76.92302710479686 38.834794725571776,-77.12352759307811 "
                                     "38.834794725571776,-77.12352759307811 38.969453670463515))")
        world_poly_3857 = photogram.reproject_geometry(world_poly_4326, crs_defs.PROJ_4326, crs_defs.PROJ_3857)
        x_points_4326 = np.array(world_poly_4326.boundary.xy[0])
        y_points_4326 = np.array(world_poly_4326.boundary.xy[1])

        x_points_3857 = np.array(world_poly_3857.boundary.xy[0])
        y_points_3857 = np.array(world_poly_3857.boundary.xy[1])
        shifted_points_4326 = photogram.shift_points_by_meters(east_shift_meters, north_shift_meters,
                                                               crs_defs.PROJ_4326, x_points_4326, y_points_4326)

        reprojected_shifted = proj_transform(crs_defs.PROJ_4326, crs_defs.PROJ_3857,
                                             shifted_points_4326[0], shifted_points_4326[1])

        assert shifted_points_4326[0][0] > x_points_4326[0]
        assert shifted_points_4326[1][0] > y_points_4326[0]

        north_shift = reprojected_shifted[0][0] - x_points_3857[0]
        east_shift = reprojected_shifted[1][0] - y_points_3857[0]

        assert np.isclose(east_shift_meters, east_shift)
        assert np.isclose(north_shift_meters, north_shift)
        print("shift by meters test passed")

    def test_omega_phi_kappa_solver(self):
        omega = np.deg2rad(5)
        phi = np.deg2rad(4)
        kappa = np.deg2rad(3)

        m_matrix = photogram.create_M_matrix(omega, phi, kappa)
        o, p, k = photogram.solve_for_omega_phi_kappa(m_matrix)
        assert np.isclose(o, omega)
        assert np.isclose(p, phi)
        assert np.isclose(k, kappa)
        print("solved for case where omega is positive, phi is positive, kappa is positive.")

        m_matrix = photogram.create_M_matrix(omega, phi, -kappa)
        o, p, k = photogram.solve_for_omega_phi_kappa(m_matrix)
        assert np.isclose(o, omega)
        assert np.isclose(p, phi)
        assert np.isclose(k, -kappa)
        print("solved for case where omega is positive, phi is positive, kappa is negative.")

        m_matrix = photogram.create_M_matrix(omega, -phi, kappa)
        o, p, k = photogram.solve_for_omega_phi_kappa(m_matrix)
        assert np.isclose(o, omega)
        assert np.isclose(p, -phi)
        assert np.isclose(k, kappa)
        print("solved for case where omega is positive, phi is negative, kappa is positive.")

        m_matrix = photogram.create_M_matrix(omega, -phi, -kappa)
        o, p, k = photogram.solve_for_omega_phi_kappa(m_matrix)
        assert np.isclose(o, omega)
        assert np.isclose(p, -phi)
        assert np.isclose(k, -kappa)
        print("solved for case where omega is positive, phi is negative, kappa is negative.")

        m_matrix = photogram.create_M_matrix(-omega, phi, kappa)
        o, p, k = photogram.solve_for_omega_phi_kappa(m_matrix)
        assert np.isclose(o, -omega)
        assert np.isclose(p, phi)
        assert np.isclose(k, kappa)
        print("solved for case where omega is negative, phi is positive, kappa is positive.")

        m_matrix = photogram.create_M_matrix(-omega, phi, -kappa)
        o, p, k = photogram.solve_for_omega_phi_kappa(m_matrix)
        assert np.isclose(o, -omega)
        assert np.isclose(p, phi)
        assert np.isclose(k, -kappa)
        print("solved for case where omega is negative, phi is positive, kappa is negative.")

        m_matrix = photogram.create_M_matrix(-omega, -phi, kappa)
        o, p, k = photogram.solve_for_omega_phi_kappa(m_matrix)
        assert np.isclose(o, -omega)
        assert np.isclose(p, -phi)
        assert np.isclose(k, kappa)
        print("solved for case where omega is negative, phi is negative, kappa is positive.")

        m_matrix = photogram.create_M_matrix(-omega, -phi, -kappa)
        o, p, k = photogram.solve_for_omega_phi_kappa(m_matrix)
        assert np.isclose(o, -omega)
        assert np.isclose(p, -phi)
        assert np.isclose(k, -kappa)
        print("solved for case where omega is negative, phi is negative, kappa is negative.")

    def test_boresight_offsets_solver(self):
        roll = np.deg2rad(4)
        pitch = np.deg2rad(2)
        yaw = np.deg2rad(30)

        roll_offset = np.deg2rad(0.3)
        pitch_offset = np.deg2rad(0.2)
        yaw_offset = np.deg2rad(3)

        offset_matrix = photogram.create_M_matrix(roll_offset, pitch_offset, yaw_offset)
        m_matrix = photogram.create_M_matrix(roll, pitch, yaw)
        m_with_offsets = m_matrix @ offset_matrix

        offsets_solved = photogram.solve_for_boresight_angle_offsets(m_matrix, m_with_offsets)
        assert np.isclose(offsets_solved, offset_matrix).all()
        print("offsets solver test passed")


if __name__ == '__main__':
    unittest.main()
