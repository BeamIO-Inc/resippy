from __future__ import division

from resippy.image_objects.earth_overhead.abstract_earth_overhead_point_calc import AbstractEarthOverheadPointCalc
import resippy.utils.photogrammetry_utils as photogram_utils
import numpy as np
from numpy import ndarray
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.pinhole_camera import PinholeCamera
from typing import Union


class FixturedCamera(AbstractEarthOverheadPointCalc):

    def __init__(self):

        self.camera = None          # type: Union[PinholeCamera]

        # all of these units are in meters and radians
        self.reference_X = None       # type: float
        self.reference_Y = None       # type: float
        self.reference_Z = None       # type: float

        self.relative_x = None      # type: float
        self.relative_y = None      # type: float
        self.relative_z = None      # type: float

        self.relative_M = None      # type: float

        self.omega = None   # type: float
        self.phi = None     # type: float
        self.kappa = None   # type: float

        self.set_projection(None)
        self._bands_coregistered = True

    def set_exterior_orientation(self, reference_x, reference_y, reference_z, omega, phi, kappa):
        reference_system = PinholeCamera.init_from_coeffs(reference_x, reference_y, reference_z, omega, phi, kappa, 1000, 5, 5)
        fixtured_M = reference_system.M @ self.relative_M
        fixtured_XYZ = reference_system.M @ [self.relative_x, self.relative_y, self.relative_z]
        omega, phi, kappa = photogram_utils.solve_for_omega_phi_kappa(fixtured_M)
        f_mm = self.camera.f*1000
        ppx_microns = self.camera.pp_x*1e6
        ppy_microns = self.camera.pp_y*1e6
        x_offset_microns = self.camera.x_0*1e6
        y_offset_microns = self.camera.y_0*1e6
        self.camera.init_from_coeffs(fixtured_XYZ[0], fixtured_XYZ[1], fixtured_XYZ[2], omega, phi, kappa, f_mm, ppx_microns, ppy_microns, x_offset_microns, y_offset_microns)

    @classmethod
    def init_from_coeffs(cls, x_meters, y_meters, z_meters, omega_radians, phi_radians, kappa_radians, f_millimeters,
                         ppx_microns, ppy_microns, x_offset_microns=0.0, y_offset_microns=0.0):
        point_calc = cls()
        point_calc.X = x_meters
        point_calc.Y = y_meters
        point_calc.Z = z_meters
        point_calc.omega = omega_radians
        point_calc.phi = phi_radians
        point_calc.kappa = kappa_radians
        point_calc.f = f_millimeters*1e-3
        point_calc.pp_x = ppx_microns*1e-6
        point_calc.pp_y = ppy_microns*1e-6
        point_calc.x_0 = x_offset_microns*1e-6
        point_calc.y_0 = y_offset_microns*1e-6

        point_calc.M = np.zeros((3, 3))

        m11 = np.cos(phi_radians)*np.cos(kappa_radians)
        m12 = np.cos(omega_radians)*np.sin(kappa_radians) + np.sin(omega_radians)*np.sin(phi_radians)*np.cos(kappa_radians)
        m13 = np.sin(omega_radians)*np.sin(kappa_radians) - np.cos(omega_radians)*np.sin(phi_radians)*np.cos(kappa_radians)

        m21 = -1.0*np.cos(phi_radians)*np.sin(kappa_radians)
        m22 = np.cos(omega_radians)*np.cos(kappa_radians) - np.sin(omega_radians)*np.sin(phi_radians)*np.sin(kappa_radians)
        m23 = np.sin(omega_radians)*np.cos(kappa_radians) + np.cos(omega_radians)*np.sin(phi_radians)*np.sin(kappa_radians)

        m31 = np.sin(phi_radians)
        m32 = -1.0*np.sin(omega_radians)*np.cos(phi_radians)
        m33 = np.cos(omega_radians)*np.cos(phi_radians)

        point_calc.m11 = m11
        point_calc.m12 = m12
        point_calc.m13 = m13

        point_calc.m21 = m21
        point_calc.m22 = m22
        point_calc.m23 = m23

        point_calc.m31 = m31
        point_calc.m32 = m32
        point_calc.m33 = m33

        point_calc.M[0, 0] = m11
        point_calc.M[0, 1] = m12
        point_calc.M[0, 2] = m13

        point_calc.M[1, 0] = m21
        point_calc.M[1, 1] = m22
        point_calc.M[1, 2] = m23

        point_calc.M[2, 0] = m31
        point_calc.M[2, 1] = m32
        point_calc.M[2, 2] = m33

        return point_calc

    def _lon_lat_alt_to_pixel_x_y_native(self,
                                         lons,  # type: ndarray
                                         lats,  # type: ndarray
                                         alts=None,  # type: ndarray
                                         band=None  # type: int
                                         ):  # type: (...) -> (ndarray, ndarray)
        pass

    def _world_to_image_space(self, world_x, world_y, world_z):
        pass

    def _pixel_x_y_alt_to_lon_lat_native(self,
                                         x_pixels,  # type: ndarray
                                         y_pixels,  # type: ndarray
                                         alts=None,  # type: ndarray
                                         band=None  # type: ndarray
                                         ):  # type: (...) -> (ndarray, ndarray)
        return None
