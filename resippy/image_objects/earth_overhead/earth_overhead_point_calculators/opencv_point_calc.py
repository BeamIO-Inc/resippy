import pyproj

from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.pinhole_camera \
    import PinholeCamera
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.abstract_earth_overhead_point_calc \
    import AbstractEarthOverheadPointCalc

import numpy as np
import cv2


class OpenCVPointCalc(AbstractEarthOverheadPointCalc):

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

        self._camera_matrix = None
        self._distortion_coeffs = None

    @classmethod
    def init_from_params(cls,
                         intrinsic_params,  # type: dict
                         extrinsic_params,  # type: dict
                         ):                 # type: (...) -> OpenCVPointCalc
        point_calc = cls()

        point_calc.set_projection(pyproj.Proj(proj='utm', zone=18, ellps='WGS84', datum='WGS84', preserve_units=True))

        point_calc.init_intrinsic(intrinsic_params['fx_pixels'], intrinsic_params['fy_pixels'],
                                  intrinsic_params['cx_pixels'], intrinsic_params['cy_pixels'],
                                  intrinsic_params['k1'], intrinsic_params['k2'], intrinsic_params['k3'],
                                  intrinsic_params['p1'], intrinsic_params['p2'])

        point_calc.init_extrinsic()

        return point_calc

    def init_intrinsic(self,
                       fx_pixels,   # type: float
                       fy_pixels,   # type: float
                       cx_pixels,   # type: float
                       cy_pixels,   # type: float
                       k1,          # type: float
                       k2,          # type: float
                       k3,          # type: float
                       p1,          # type: float
                       p2           # type: float
                       ):           # type: (...) -> None
        self._fx_pixels = -fx_pixels
        self._fy_pixels = -fy_pixels
        self._cx_pixels = cx_pixels
        self._cy_pixels = cy_pixels
        self._k1 = k1
        self._k2 = k2
        self._k3 = k3
        self._p1 = p1
        self._p2 = p2

        self._camera_matrix = np.array([[self._fx_pixels, 0, self._cx_pixels],
                                        [0, self._fy_pixels, self._cy_pixels],
                                        [0, 0, 1]], dtype=np.float64)

        self._distortion_coeffs = np.array([self._k1, self._k2, self._p1, self._p2, self._k3], dtype=np.float64)

    def init_extrinsic(self
                       ):
        pass

    def _lon_lat_alt_to_pixel_x_y_native(self,
                                         lons,          # type: np.ndarray
                                         lats,          # type: np.ndarray
                                         alts=None,     # type: np.ndarray
                                         band=None      # type: int
                                         ):             # type: (...) -> (np.ndarray, np.ndarray)
        # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        trans_xyz = np.ones((3, len(lons)))
        trans_xyz[0, :] = lons - self._trans[0]
        trans_xyz[1, :] = lats - self._trans[1]
        trans_xyz[2, :] = alts - self._trans[2]

        cam_coords = np.matmul(self._rot, trans_xyz)

        x_prime = cam_coords[0, :] / cam_coords[2, :]
        y_prime = cam_coords[1, :] / cam_coords[2, :]
        r_squared = (x_prime * x_prime) + (y_prime * y_prime)
        r_fourth = r_squared * r_squared
        r_sixth = r_squared * r_squared * r_squared
        radial_distortion = 1.0 + (self._k1 * r_squared) + (self._k2 * r_fourth) + (self._k3 * r_sixth)

        x_double_prime = (x_prime * radial_distortion) + (2.0 * self._p1 * x_prime * y_prime) + \
                         (self._p2 * (r_squared + 2.0 * x_prime * x_prime))
        y_double_prime = (y_prime * radial_distortion) + (self._p1 * (r_squared + 2.0 * y_prime * y_prime)) + \
                         (2.0 * self._p2 * x_prime * y_prime)

        u = -self._fx_pixels * x_double_prime + self._cx_pixels
        v = -self._fy_pixels * y_double_prime + self._cy_pixels

        return u, v

        # object_points = np.ones((len(lons), 3), dtype=np.float64)
        # object_points[:, 0] = lons
        # object_points[:, 1] = lats
        # object_points[:, 2] = alts
        #
        # rot_vec, __ = cv2.Rodrigues(self._rot)
        #
        # print(rot_vec)
        #
        # image_points, __ = cv2.projectPoints(object_points, rot_vec, self._trans, self._camera_matrix,
        #                                      self._distortion_coeffs)
        #
        # return image_points[:, :, 0], image_points[:, :, 1]

    def _pixel_x_y_alt_to_lon_lat_native(self,
                                         pixel_xs,      # type: np.ndarray
                                         pixel_ys,      # type: np.ndarray
                                         alts=None,     # type: np.ndarray
                                         band=None      # type: np.ndarray
                                         ):             # type: (...) -> (np.ndarray, np.ndarray)
        pass
