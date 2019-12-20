from __future__ import division

import unittest
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.pinhole_camera import PinholeCamera
import numpy as np
import resippy.utils.photogrammetry_utils as photogram_utils


class TestPointCalcs(unittest.TestCase):

    def test_rotation_matrix(self):
        omega = np.deg2rad(1.5)
        phi = np.deg2rad(2.3)
        kappa = np.deg2rad(3.4)
        camera = PinholeCamera()
        camera.init_pinhole_from_coeffs(0, 0, 0, omega, phi, kappa, 100)

        m_omega = np.zeros((3, 3))
        m_phi = np.zeros((3, 3))
        m_kappa = np.zeros((3, 3))

        m_omega[0, 0] = 1.0
        m_omega[1, 1] = np.cos(omega)
        m_omega[1, 2] = np.sin(omega)
        m_omega[2, 1] = -1*np.sin(omega)
        m_omega[2, 2] = np.cos(omega)

        m_phi[0, 0] = np.cos(phi)
        m_phi[0, 2] = -1.0*np.sin(phi)
        m_phi[1, 1] = 1.0
        m_phi[2, 0] = np.sin(phi)
        m_phi[2, 2] = np.cos(phi)

        m_kappa[0, 0] = np.cos(kappa)
        m_kappa[0, 1] = np.sin(kappa)
        m_kappa[1, 0] = -1.0*np.sin(kappa)
        m_kappa[1, 1] = np.cos(kappa)
        m_kappa[2, 2] = 1.0

        m_1 = np.matmul(m_kappa, np.matmul(m_phi, m_omega))
        m_2 = m_kappa @ m_phi @ m_omega

        assert np.isclose(m_1, m_2).all()
        assert np.isclose(m_2, camera.M).all()

        print("confirming rotation order for standard photogrammetric operations:")
        print("the correct order of operations for:")
        print("(first rotation) omega, then (second rotation) phi, then (third rotation) kapa is:")
        print("m_kappa @ m_phi @ m_omega")
        print("or")
        print("np.matmul(m_kappa, np.matmul(m_phi, m_omega))")

    def test_projection(self):
        camera = PinholeCamera()
        camera.init_pinhole_from_coeffs(0, 0, 100, 0, 0, 0, 100)
        x, y = camera.world_to_image_plane(100, 0, 0)
        assert x == camera.f
        stop = 1

    def test_relative_orientation(self):
        omega = np.deg2rad(1.5)
        phi = np.deg2rad(2.3)
        kappa = np.deg2rad(3.4)
        camera = PinholeCamera()
        camera.init_pinhole_from_coeffs(0, 0, 0, omega, phi, kappa, 100)

        omega_1, phi_1, kappa_1 = photogram_utils.solve_for_omega_phi_kappa(camera.M)

        assert np.isclose(omega, omega_1, atol=1.e-15)
        assert np.isclose(phi, phi_1, atol=1.e-15)
        assert np.isclose(kappa, kappa_1, atol=1.e-15)

        print("solved values for omega, phi and kappa are within 1.e-15")


if __name__ == '__main__':
    unittest.main()
