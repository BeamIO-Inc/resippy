from __future__ import division

from resippy.image_objects.earth_overhead.abstract_earth_overhead_point_calc import AbstractEarthOverheadPointCalc
from numpy import ndarray
from resippy.utils import photogrammetry_utils


class PinholeCamera(AbstractEarthOverheadPointCalc):
    """
    This is an idealized pinhole camera model.
    """
    # TODO find the photogrammetry book that was used to implement all of these calculations and cite it here.

    def __init__(self):
        # all of these units are in meters and radians
        self.X = None       # type: float
        self.Y = None       # type: float
        self.Z = None       # type: float
        self.omega = None   # type: float
        self.phi = None     # type: float
        self.kappa = None   # type: float
        self.f = None       # type: float
        self.M = None       # type: ndarray

        self.m11 = None
        self.m12 = None
        self.m13 = None

        self.m21 = None
        self.m22 = None
        self.m23 = None

        self.m31 = None
        self.m32 = None
        self.m33 = None

        self.pp_x = None
        self.pp_y = None

        self.x_0 = None
        self.y_0 = None

        self.set_projection(None)
        self._bands_coregistered = True

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

        point_calc.M = photogrammetry_utils.create_M_matrix(omega_radians, phi_radians, kappa_radians)

        point_calc.m11 = point_calc.M[0, 0]
        point_calc.m12 = point_calc.M[0, 1]
        point_calc.m13 = point_calc.M[0, 2]

        point_calc.m21 = point_calc.M[1, 0]
        point_calc.m22 = point_calc.M[1, 1]
        point_calc.m23 = point_calc.M[1, 2]

        point_calc.m31 = point_calc.M[2, 0]
        point_calc.m32 = point_calc.M[2, 1]
        point_calc.m33 = point_calc.M[2, 2]

        return point_calc

    def _lon_lat_alt_to_pixel_x_y_native(self,
                                         lons,  # type: ndarray
                                         lats,  # type: ndarray
                                         alts=None,  # type: ndarray
                                         band=None  # type: int
                                         ):  # type: (...) -> (ndarray, ndarray)
        pass

    def _world_to_image_space(self, world_x, world_y, world_z):
        x = -1.0*self.f*(
                (self.m11*(world_x - self.X) + self.m12*(world_y - self.Y) + self.m13*(world_z - self.Z)) /
                (self.m31*(world_x - self.X) + self.m32*(world_y - self.Y) + self.m33*(world_z - self.Z))
        ) + self.x_0
        y = -1.0*self.f*(
                (self.m21*(world_x - self.X) + self.m22*(world_y - self.Y) + self.m23*(world_z - self.Z)) /
                (self.m31*(world_x - self.X) + self.m32*(world_y - self.Y) + self.m33*(world_z - self.Z))
        ) + self.y_0
        return x, y

    def _pixel_x_y_alt_to_lon_lat_native(self,
                                         x_pixels,  # type: ndarray
                                         y_pixels,  # type: ndarray
                                         alts=None,  # type: ndarray
                                         band=None  # type: ndarray
                                         ):  # type: (...) -> (ndarray, ndarray)
        return None

