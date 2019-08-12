import pyproj

from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.pinhole_camera \
    import PinholeCamera
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.abstract_earth_overhead_point_calc \
    import AbstractEarthOverheadPointCalc

import numpy as np


class PhysicalModelPointCalc(PinholeCamera, AbstractEarthOverheadPointCalc):

    def __init__(self):
        self._fx_pixels = 0
        self._fy_pixels = 0
        self._cx_pixels = 0
        self._cy_pixels = 0
        self._k1 = 0.0
        self._k2 = 0.0
        self._k3 = 0.0
        self._p1 = 0.0
        self._p2 = 0.0

    @classmethod
    def init_from_params_and_center(cls,
                                    intrinsic_params,   # type: dict
                                    extrinsic_params,   # type: dict
                                    center_lon,         # type: float
                                    center_lat,         # type: float
                                    ):                  # type: (...) -> PhysicalModelPointCalc
        point_calc = cls()

        point_calc.set_approximate_lon_lat_center(center_lon, center_lat)
        point_calc.set_projection(pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84'))

        point_calc.init_pinhole_from_coeffs(extrinsic_params['X'], extrinsic_params['Y'], extrinsic_params['Z'],
                                            extrinsic_params['omega'], extrinsic_params['phi'],
                                            extrinsic_params['kappa'], extrinsic_params['focal_length'])

        point_calc.init_physical_from_coeffs(intrinsic_params['fx_pixels'], intrinsic_params['fy_pixels'],
                                             intrinsic_params['cx_pixels'], intrinsic_params['cy_pixels'],
                                             intrinsic_params['k1'], intrinsic_params['k2'], intrinsic_params['k3'],
                                             intrinsic_params['p1'], intrinsic_params['p2'])

        return point_calc

    def init_physical_from_coeffs(self,
                                  fx_pixels,    # type: float
                                  fy_pixels,    # type: float
                                  cx_pixels,    # type: float
                                  cy_pixels,    # type: float
                                  k1,           # type: float
                                  k2,           # type: float
                                  k3,           # type: float
                                  p1,           # type: float
                                  p2            # type: float
                                  ):            # type: (...) -> None
        self._fx_pixels = fx_pixels
        self._fy_pixels = fy_pixels
        self._cx_pixels = cx_pixels
        self._cy_pixels = cy_pixels
        self._k1 = k1
        self._k2 = k2
        self._k3 = k3
        self._p1 = p1
        self._p2 = p2

    def _lon_lat_alt_to_pixel_x_y_native(self,
                                         lons,          # type: np.ndarray
                                         lats,          # type: np.ndarray
                                         alts=None,     # type: np.ndarray
                                         band=None      # type: int
                                         ):             # type: (...) -> (np.ndarray, np.ndarray)
        # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
        x, y, z = pyproj.transform(self.get_projection(), ecef, lons, lats, alts, radians=False)

        x_prime = x / z
        y_prime = y / z
        r_squared = (x_prime * x_prime) + (y_prime * y_prime)
        r_fourth = r_squared * r_squared
        r_sixth = r_squared * r_squared * r_squared
        radial_distortion = 1.0 + (self._k1 * r_squared) + (self._k2 * r_fourth) + (self._k3 * r_sixth)

        x_double_prime = (x_prime * radial_distortion) + (2.0 * self._p1 * x_prime * y_prime) + \
                         (self._p2 * (r_squared + 2.0 * x_prime * x_prime))
        y_double_prime = (y_prime * radial_distortion) + (self._p1 * (r_squared + 2 * y_prime * y_prime)) + \
                         (2.0 * self._p2 * x_prime * y_prime)

        u = self._fx_pixels * x_double_prime + self._cx_pixels
        v = self._fy_pixels + y_double_prime + self._cy_pixels

        return u, v

    def _pixel_x_y_alt_to_lon_lat_native(self,
                                         pixel_xs,      # type: np.ndarray
                                         pixel_ys,      # type: np.ndarray
                                         alts=None,     # type: np.ndarray
                                         band=None      # type: np.ndarray
                                         ):             # type: (...) -> (np.ndarray, np.ndarray)
        # TODO
        pass
