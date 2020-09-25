from __future__ import division

from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.supporting_classes.pinhole_camera import PinholeCamera
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.abstract_earth_overhead_point_calc import AbstractEarthOverheadPointCalc
from resippy.photogrammetry import crs_defs
from resippy.utils import proj_utils
from pyproj import transform
from pyproj import Proj
from resippy.utils.units import ureg
from numpy import ndarray
import numpy


class IdealPinholeFpaLocalUtmPointCalc(AbstractEarthOverheadPointCalc):

    def __init__(self):
        super(IdealPinholeFpaLocalUtmPointCalc, self).__init__()
        self._pinhole_camera = None                 # type: PinholeCamera
        self._lon_lat_center_approximate = None     # type: tuple
        self._pixel_pitch_x_meters = None           # type: float
        self._pixel_pitch_y_meters = None           # type: float
        self._npix_x = None                         # type: int
        self._npix_y = None                         # type: int
        self._flip_x = False                        # type: bool
        self._flip_y = False                        # type: bool

    def _pixel_x_y_alt_to_lon_lat_native(self, pixel_xs, pixel_ys, alts, band=None):
        image_plane_xs, image_plane_ys = self.pixel_coords_to_image_plane_coords(pixel_xs, pixel_ys)
        return self._pinhole_camera.image_to_world_plane(image_plane_xs, image_plane_ys, alts)

    def _lon_lat_alt_to_pixel_x_y_native(self,
                                         lons,          # type: ndarray
                                         lats,          # type: ndarray
                                         alts,          # type: ndarray
                                         band=None      # type: int
                                         ):
        fpa_coords_meters_x, fpa_coords_meters_y = self._pinhole_camera.world_to_image_plane(lons, lats, alts)
        return self.image_plane_coords_to_pixel_coords(fpa_coords_meters_x, fpa_coords_meters_y)

    def image_plane_coords_to_pixel_coords(self,
                                           fpa_coords_meters_x,  # type: ndarray
                                           fpa_coords_meters_y,  # type: ndarray
                                           ):
        half_fpa_x_meters = (self._npix_x * self._pixel_pitch_x_meters) / 2.0
        half_fpa_y_meters = (self._npix_y * self._pixel_pitch_y_meters) / 2.0
        if self._flip_x:
            fpa_coords_meters_x = -1.0 * fpa_coords_meters_x
        if self._flip_y:
            fpa_coords_meters_y = -1.0 * fpa_coords_meters_y
        fpa_coords_pixels_x = (fpa_coords_meters_x + half_fpa_x_meters - self._pixel_pitch_x_meters/2) / self._pixel_pitch_x_meters
        fpa_coords_pixels_y = (fpa_coords_meters_y + half_fpa_y_meters - self._pixel_pitch_y_meters/2) / self._pixel_pitch_y_meters
        return fpa_coords_pixels_x, fpa_coords_pixels_y

    def pixel_coords_to_image_plane_coords(self, pixel_coords_x, pixel_coords_y):
        half_fpa_x_meters = (self._npix_x * self._pixel_pitch_x_meters) / 2.0
        half_fpa_y_meters = (self._npix_y * self._pixel_pitch_y_meters) / 2.0
        if self._flip_x:
            pixel_coords_x = -1.0 * pixel_coords_x
        if self._flip_y:
            pixel_coords_y = -1.0 * pixel_coords_y
        image_plane_coords_x = pixel_coords_x * self._pixel_pitch_x_meters - half_fpa_x_meters + self._pixel_pitch_x_meters/2
        image_plane_coords_y = pixel_coords_y * self._pixel_pitch_y_meters - half_fpa_y_meters + self._pixel_pitch_y_meters/2
        return image_plane_coords_x, image_plane_coords_y

    @classmethod
    def init_from_wgs84_params(cls,
                               sensor_lon_decimal_degrees,              # type: float
                               sensor_lat_decimal_degrees,              # type: float
                               sensor_altitude,                         # type: float
                               omega,                                   # type: float
                               phi,                                     # type: float
                               kappa,                                   # type: float
                               npix_x,                                  # type: int
                               npix_y,                                  # type: int
                               pixel_pitch_x,                           # type: float
                               pixel_pitch_y,                           # type: float
                               focal_length,                            # type: float
                               alt_units='meters',                      # type: str
                               omega_units='radians',                   # type: str
                               phi_units='radians',                     # type: str
                               kappa_units='radians',                   # type: str
                               pixel_pitch_x_units='micrometer',        # type: str
                               pixel_pitch_y_units='micrometer',        # type: str
                               focal_length_units='mm',                 # type: str
                               flip_x=False,                            # type: bool
                               flip_y=False,                            # type: bool
                               ):                                       # type: (...) -> IdealPinholeFpaLocalUtmPointCalc
        """

        :param sensor_lon_decimal_degrees:
        :param sensor_lat_decimal_degrees:
        :param sensor_altitude:
        :param omega:
        :param phi:
        :param kappa:
        :param npix_x:
        :param npix_y:
        :param pixel_pitch_x:
        :param pixel_pitch_y:
        :param focal_length:
        :param alt_units:
        :param omega_units:
        :param phi_units:
        :param kappa_units:
        :param pixel_pitch_x_units:
        :param pixel_pitch_y_units:
        :param focal_length_units:
        :param flip_x:
        :param flip_y:
        :return:

        For now the altitude must be relative to the DEM used for orthorectification.
        For example, if the DEM is relative to the geoid, then sensor_altitude must also be relative to the geoid too.
        """
        native_proj = proj_utils.decimal_degrees_to_local_utm_proj(sensor_lon_decimal_degrees, sensor_lat_decimal_degrees)
        proj_4326 = crs_defs.PROJ_4326

        sensor_x_local, sensor_y_local = transform(proj_4326,
                                                   native_proj,
                                                   sensor_lon_decimal_degrees,
                                                   sensor_lat_decimal_degrees)

        utm_point_calc = IdealPinholeFpaLocalUtmPointCalc()

        pinhole_camera = PinholeCamera()
        pinhole_camera.init_pinhole_from_coeffs(sensor_x_local,
                                                sensor_y_local,
                                                sensor_altitude,
                                                omega,
                                                phi,
                                                kappa,
                                                focal_length,
                                                x_units='meters',
                                                y_units='meters',
                                                z_units=alt_units,
                                                omega_units=omega_units,
                                                phi_units=phi_units,
                                                kappa_units=kappa_units,
                                                focal_length_units=focal_length_units)

        utm_point_calc.set_projection(native_proj)

        pp_x_meters = pixel_pitch_x * ureg.parse_expression(pixel_pitch_x_units)
        pp_y_meters = pixel_pitch_y * ureg.parse_expression(pixel_pitch_y_units)

        pp_x_meters = pp_x_meters.to('meters').magnitude
        pp_y_meters = pp_y_meters.to('meters').magnitude

        utm_point_calc._pixel_pitch_x_meters = pp_x_meters
        utm_point_calc._pixel_pitch_y_meters = pp_y_meters
        utm_point_calc.set_approximate_lon_lat_center(sensor_x_local, sensor_y_local)
        utm_point_calc._npix_x = npix_x
        utm_point_calc._npix_y = npix_y
        utm_point_calc._flip_x = flip_x
        utm_point_calc._flip_y = flip_y
        utm_point_calc._pinhole_camera = pinhole_camera
        return utm_point_calc

    @classmethod
    def init_from_local_params(cls,
                               sensor_lon,                              # type: float
                               sensor_lat,                              # type: float
                               sensor_altitude,                         # type: float
                               native_proj,                             # type: Proj
                               omega,                                   # type: float
                               phi,                                     # type: float
                               kappa,                                   # type: float
                               npix_x,                                  # type: int
                               npix_y,                                  # type: int
                               pixel_pitch_x,                           # type: float
                               pixel_pitch_y,                           # type: float
                               focal_length,                            # type: float
                               alt_units='meters',                      # type: str
                               omega_units='radians',                   # type: str
                               phi_units='radians',                     # type: str
                               kappa_units='radians',                   # type: str
                               pixel_pitch_x_units='micrometer',        # type: str
                               pixel_pitch_y_units='micrometer',        # type: str
                               focal_length_units='mm',                 # type: str
                               flip_x=False,                            # type: bool
                               flip_y=False,                            # type: bool
                               ):                                       # type: (...) -> IdealPinholeFpaLocalUtmPointCalc

        utm_point_calc = IdealPinholeFpaLocalUtmPointCalc()

        pinhole_camera = PinholeCamera()
        pinhole_camera.init_pinhole_from_coeffs(sensor_lon,
                                                sensor_lat,
                                                sensor_altitude,
                                                omega,
                                                phi,
                                                kappa,
                                                focal_length,
                                                x_units='meters',
                                                y_units='meters',
                                                z_units=alt_units,
                                                omega_units=omega_units,
                                                phi_units=phi_units,
                                                kappa_units=kappa_units,
                                                focal_length_units=focal_length_units)

        utm_point_calc.set_projection(native_proj)

        pp_x_meters = pixel_pitch_x * ureg.parse_expression(pixel_pitch_x_units)
        pp_y_meters = pixel_pitch_y * ureg.parse_expression(pixel_pitch_y_units)

        pp_x_meters = pp_x_meters.to('meters').magnitude
        pp_y_meters = pp_y_meters.to('meters').magnitude

        utm_point_calc._pixel_pitch_x_meters = pp_x_meters
        utm_point_calc._pixel_pitch_y_meters = pp_y_meters
        utm_point_calc.set_approximate_lon_lat_center(sensor_lon, sensor_lat)
        utm_point_calc._npix_x = npix_x
        utm_point_calc._npix_y = npix_y
        utm_point_calc._flip_x = flip_x
        utm_point_calc._flip_y = flip_y
        utm_point_calc._pinhole_camera = pinhole_camera
        return utm_point_calc

    def compute_ifov_x(self, output_units="microradians"):
        ifov = numpy.arctan(self._pixel_pitch_x_meters/self._pinhole_camera.f) * ureg.parse_expression('radians')
        return ifov.to(output_units).magnitude

    def compute_ifov_y(self, output_units="microradians"):
        ifov = numpy.arctan(self._pixel_pitch_y_meters/self._pinhole_camera.f) * ureg.parse_expression('radians')
        return ifov.to(output_units).magnitude
