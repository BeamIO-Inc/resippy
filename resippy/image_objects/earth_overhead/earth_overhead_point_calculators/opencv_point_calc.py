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
    def init_from_params_and_center(cls,
                                    intrinsic_params,   # type: dict
                                    extrinsic_params,   # type: dict
                                    center_lon,         # type: float
                                    center_lat,         # type: float
                                    ):                  # type: (...) -> OpenCVPointCalc
        point_calc = cls()

        point_calc.set_approximate_lon_lat_center(center_lon, center_lat)
        point_calc.set_projection(pyproj.Proj(proj='utm', zone=18, ellps='WGS84', datum='WGS84'))

        point_calc.init_coeffs(intrinsic_params['fx_pixels'], intrinsic_params['fy_pixels'],
                               intrinsic_params['cx_pixels'], intrinsic_params['cy_pixels'],
                               intrinsic_params['k1'], intrinsic_params['k2'], intrinsic_params['k3'],
                               intrinsic_params['p1'], intrinsic_params['p2'])

        trans = [[334265.746789, 4748702.207249, 262.700231], [334265.717233, 4748702.203111, 262.697172],
                 [334265.720081, 4748702.182314, 262.697784], [334265.749637, 4748702.186452, 262.700843],
                 [334265.733435, 4748702.194782, 262.699007]]

        # trans = [[18.74678864, -41.79275101,  44.70023088], [18.71723325, -41.79688858,  44.69717154],
        #          [18.72008132, -41.81768554,  44.6977838], [18.74963671, -41.81354797,  44.70084314],
        #          [18.73343498, -41.80521827,  44.69900734]]

        # rots = [[-0.788417, -5.994801, 8.313844], [-0.858048, -6.027922, 7.83822], [-1.051687, -6.160215, 8.401133],
        #         [-0.899741, -6.121322, 8.239459], [-0.863627, -6.121558, 8.376414]]

        rots = [[[0.98407973, 0.14600357, 0.10134124],
                 [0.14380455, -0.98918941, 0.02871527],
                 [0.10443822, -0.01368478, -0.99443722]],
                [[0.98517958, 0.13791905, 0.10197805],
                 [0.13562239, -0.99033154, 0.029155],
                 [0.10501311, -0.0148924, -0.99435932]],
                [[0.98355712, 0.14802643, 0.10345803],
                 [0.14525895, -0.98881503, 0.0338329],
                 [0.10730902, -0.01824838, -0.99405823]],
                [[0.98403495, 0.14495006, 0.10327],
                 [0.14249345, -0.98931578, 0.03082062],
                 [0.10663409, -0.01561327, -0.99417574]],
                [[0.98369111, 0.14724939, 0.10329292],
                 [0.14484511, -0.98898585, 0.03044462],
                 [0.10663818, -0.01498663, -0.99418494]]]

        point_calc._trans = np.array(trans[intrinsic_params['band_number'] - 1], dtype=np.float64)
        point_calc._rot = np.array(rots[intrinsic_params['band_number'] - 1], dtype=np.float64)

        return point_calc

    def init_coeffs(self,
                    fx_pixels,     # type: float
                    fy_pixels,     # type: float
                    cx_pixels,     # type: float
                    cy_pixels,     # type: float
                    k1,            # type: float
                    k2,            # type: float
                    k3,            # type: float
                    p1,            # type: float
                    p2             # type: float
                    ):             # type: (...) -> None
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
