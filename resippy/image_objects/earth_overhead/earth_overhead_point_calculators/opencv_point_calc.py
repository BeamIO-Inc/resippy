import pyproj
import numpy as np

from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.abstract_earth_overhead_point_calc \
    import AbstractEarthOverheadPointCalc
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.fixtured_camera import FixturedCamera


class OpenCVPointCalc(AbstractEarthOverheadPointCalc):

    def __init__(self):
        super(OpenCVPointCalc, self).__init__()

        # intrinsic params
        self._fx_pixels = 0
        self._fy_pixels = 0
        self._cx_pixels = 0
        self._cy_pixels = 0
        self._k1 = 0.0
        self._k2 = 0.0
        self._k3 = 0.0
        self._p1 = 0.0
        self._p2 = 0.0
        self._pixel_pitch_microns = 0.0

        self._camera_matrix = None
        self._distortion_coeffs = None

        # orientation params
        self._x_meters = 0.0
        self._y_meters = 0.0
        self._z_meters = 0.0
        self._omega_radians = 0.0
        self._phi_radians = 0.0
        self._kappa_radians = 0.0

        # offsets
        self._x_offset_meters = 0.0
        self._y_offset_meters = 0.0
        self._z_offset_meters = 0.0
        self._omega_offset_radians = 0.0
        self._phi_offset_radians = 0.0
        self._kappa_offset_radians = 0.0

        # fixture
        self._fixture = FixturedCamera()

    @classmethod
    def init_from_params(cls,
                         params,    # type: dict
                         ):         # type: (...) -> OpenCVPointCalc
        point_calc = cls()

        point_calc.set_projection(pyproj.Proj(proj='utm', zone=18, ellps='WGS84', datum='WGS84', preserve_units=True))

        point_calc.init_intrinsic(params['fx_pixels'], params['fy_pixels'], params['cx_pixels'], params['cy_pixels'],
                                  params['k1'], params['k2'], params['k3'], params['p1'], params['p2'],
                                  params['pixel_pitch_microns'])

        point_calc.init_offsets(params['x_offset_meters'], params['y_offset_meters'], params['z_offset_meters'],
                                params['omega_offset_radians'], params['phi_offset_radians'],
                                params['kappa_offset_radians'])

        return point_calc

    def init_intrinsic(self,
                       fx_pixels,           # type: float
                       fy_pixels,           # type: float
                       cx_pixels,           # type: float
                       cy_pixels,           # type: float
                       k1,                  # type: float
                       k2,                  # type: float
                       k3,                  # type: float
                       p1,                  # type: float
                       p2,                  # type: float
                       pixel_pitch_microns  # type: float
                       ):                   # type: (...) -> None
        self._fx_pixels = -fx_pixels
        self._fy_pixels = -fy_pixels
        self._cx_pixels = cx_pixels
        self._cy_pixels = cy_pixels
        self._k1 = k1
        self._k2 = k2
        self._k3 = k3
        self._p1 = p1
        self._p2 = p2
        self._pixel_pitch_microns = pixel_pitch_microns

        self._camera_matrix = np.array([[self._fx_pixels, 0, self._cx_pixels],
                                        [0, self._fy_pixels, self._cy_pixels],
                                        [0, 0, 1]], dtype=np.float64)

        self._distortion_coeffs = np.array([self._k1, self._k2, self._p1, self._p2, self._k3], dtype=np.float64)

    def init_extrinsic(self,
                       x_meters,        # type: float
                       y_meters,        # type: float
                       z_meters,        # type: float
                       omega_radians,   # type: float
                       phi_radians,     # type: float
                       kappa_radians    # type: float
                       ):               # type: (...) -> None
        self._x_meters = x_meters
        self._y_meters = y_meters
        self._z_meters = z_meters
        self._omega_radians = omega_radians
        self._phi_radians = phi_radians
        self._kappa_radians = kappa_radians

        self._fixture.set_fixture_xyz(x_meters, y_meters, z_meters)
        self._fixture.set_fixture_orientation_by_roll_pitch_yaw(omega_radians, phi_radians, kappa_radians)

    def init_offsets(self,
                     x_offset_meters,       # type: float
                     y_offset_meters,       # type: float
                     z_offset_meters,       # type: float
                     omega_offset_radians,  # type: float
                     phi_offset_radians,    # type: float
                     kappa_offset_radians   # type: float
                     ):                     # type: (...) -> None
        self._x_offset_meters = x_offset_meters
        self._y_offset_meters = y_offset_meters
        self._z_offset_meters = z_offset_meters
        self._omega_offset_radians = omega_offset_radians
        self._phi_offset_radians = phi_offset_radians
        self._kappa_offset_radians = kappa_offset_radians

        self._fixture.set_relative_camera_xyz(x_offset_meters, y_offset_meters, z_offset_meters)
        self._fixture.set_boresight_matrix_from_camera_relative_rpy_params(omega_offset_radians, phi_offset_radians,
                                                                           kappa_offset_radians)

    def _lon_lat_alt_to_pixel_x_y_native(self,
                                         lons,          # type: np.ndarray
                                         lats,          # type: np.ndarray
                                         alts=None,     # type: np.ndarray
                                         band=None      # type: int
                                         ):             # type: (...) -> (np.ndarray, np.ndarray)
        # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        camera_rot = self._fixture.get_camera_absolute_M_matrix()
        camera_xyz = self._fixture.get_camera_absolute_xyz()

        trans_xyz = np.ones((3, len(lons)))
        trans_xyz[0, :] = lons - camera_xyz[0]
        trans_xyz[1, :] = lats - camera_xyz[1]
        trans_xyz[2, :] = alts - camera_xyz[2]

        cam_coords = np.matmul(camera_rot, trans_xyz)

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
        v = self._fy_pixels * y_double_prime + self._cy_pixels

        return u, v

    def _pixel_x_y_alt_to_lon_lat_native(self,
                                         pixel_xs,      # type: np.ndarray
                                         pixel_ys,      # type: np.ndarray
                                         alts=None,     # type: np.ndarray
                                         band=None      # type: np.ndarray
                                         ):             # type: (...) -> (np.ndarray, np.ndarray)
        pass
