from __future__ import division

import numpy
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.supporting_classes.pinhole_camera import PinholeCamera
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.supporting_classes.fixtured_camera import FixturedCamera
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.abstract_earth_overhead_point_calc import AbstractEarthOverheadPointCalc
from resippy.photogrammetry import crs_defs
from resippy.utils import proj_utils
from resippy.utils.image_utils import image_utils
from resippy.utils.units import ureg
from pyproj import transform
from pyproj import Proj
from numpy import ndarray


class FpaDistortionMappedPointCalc(AbstractEarthOverheadPointCalc):
    def __init__(self):
        self._lon_lat_center_approximate = None     # type: tuple
        self._fixture = FixturedCamera()            # type: FixturedCamera
        self._focal_length_meters = None            # type: float
        self._native_proj = None        # type: Proj
        self._undistorted_x_grid = None     # type: ndarray
        self._undistorted_y_grid = None     # type: ndarray

    def _lon_lat_alt_to_pixel_x_y_native(self, lons, lats, alts, band=None):
        pass

    def _pixel_x_y_alt_to_lon_lat_native(self,
                                         pixel_xs,  # type: ndarray
                                         pixel_ys,  # type: ndarray
                                         alts,      # type: ndarray
                                         band=None,  # type: int
                                         ):
        if len(pixel_xs.shape) == 2:
            input_ny, input_nx = pixel_xs.shape
        x_ravel = pixel_xs.ravel()
        y_ravel = pixel_ys.ravel()
        undistorted_xs = self._undistorted_x_grid[[y_ravel], [x_ravel]]
        undistorted_ys = self._undistorted_y_grid[[y_ravel], [x_ravel]]
        if len(pixel_xs.shape) == 2:
            undistorted_xs = numpy.reshape(undistorted_xs, (input_ny, input_nx))
            undistorted_ys = numpy.reshape(undistorted_ys, (input_ny, input_nx))

        camera_x, camera_y, camera_z = self._fixture.get_camera_absolute_xyz()
        camera_m_matrix = self._fixture.get_camera_absolute_M_matrix()
        pinhole_camera = PinholeCamera()
        pinhole_camera.init_pinhole_from_m_matrix(camera_x,
                                                  camera_y,
                                                  camera_z,
                                                  camera_m_matrix,
                                                  self._focal_length_meters,
                                                  x_units='meters',
                                                  y_units='meters',
                                                  z_units='meters',
                                                  focal_length_units='meters')
        return pinhole_camera.image_to_world_plane(undistorted_xs, undistorted_ys, alts)

    @classmethod
    def create_pinhole_model(cls,
                             npix_x,  # type: ndarray
                             npix_y,  # type: ndarray
                             pixel_pitch_x,  # type: int
                             pixel_pitch_y,  # type: int
                             pixel_pitch_units='micrometers',  # type: str
                             ):
        model = cls()
        x_spacing = (ureg.parse_units(pixel_pitch_units) * pixel_pitch_x).to('meters').magnitude
        y_spacing = (ureg.parse_units(pixel_pitch_units) * pixel_pitch_y).to('meters').magnitude
        image_plane_grid = image_utils.create_image_plane_grid(npix_x, npix_y, x_spacing=x_spacing, y_spacing=y_spacing)
        model._undistorted_x_grid = image_plane_grid[0]
        model._undistorted_y_grid = image_plane_grid[1]
        return model

    def set_focal_length(self, focal_length, units='mm'):
        self._focal_length_meters = (focal_length * ureg.parse_units(units)).to('meters').magnitude

    def set_xyz_using_wgs84_coords(self,
                                   sensor_lon_decimal_degrees,  # type: float
                                   sensor_lat_decimal_degrees,  # type: float
                                   sensor_altitude,  # type: float
                                   alt_units='meters',  # type: str
                                   ):  # type: (...) -> IdealPinholeFpaLocalUtmPointCalc

        native_proj = proj_utils.decimal_degrees_to_local_utm_proj(sensor_lon_decimal_degrees,
                                                                   sensor_lat_decimal_degrees)
        proj_4326 = crs_defs.PROJ_4326

        sensor_x_local, sensor_y_local = transform(proj_4326,
                                                   native_proj,
                                                   sensor_lon_decimal_degrees,
                                                   sensor_lat_decimal_degrees)

        self.set_projection(native_proj)
        self.set_xyz_using_local_coords(sensor_x_local, sensor_y_local, sensor_altitude, alt_units=alt_units)

    def set_xyz_using_local_coords(self,
                                   sensor_lon,  # type: float
                                   sensor_lat,  # type: float
                                   sensor_altitude,  # type: float
                                   alt_units='meters',  # type: str
                                   ):  # type: (...) -> IdealPinholeFpaLocalUtmPointCalc
        alt_meters = (sensor_altitude * ureg.parse_units(alt_units)).to('meters').magnitude
        self._fixture.set_fixture_xyz(sensor_lon, sensor_lat, alt_meters)

    def set_roll_pitch_yaw(self,
                           roll,
                           pitch,
                           yaw,
                           units='radians',
                           order='rpy'):
        self._fixture.set_fixture_orientation_by_roll_pitch_yaw(roll,
                                                                pitch,
                                                                yaw,
                                                                roll_units=units,
                                                                pitch_units=units,
                                                                yaw_units=units,
                                                                order=order)

    def set_boresight_roll_pitch_yaw_offsets(self,
                                             boresight_roll,
                                             boresight_pitch,
                                             boresight_yaw,
                                             units='radians',
                                             order='rpy',
                                             ):
        self._fixture.set_boresight_matrix_from_camera_relative_rpy_params(boresight_roll,
                                                                           boresight_pitch,
                                                                           boresight_yaw,
                                                                           roll_units=units,
                                                                           pitch_units=units,
                                                                           yaw_units=units,
                                                                           order=order)

    def set_mounting_position_on_fixture(self, x, y, z, units='meters'):
        self._fixture.set_relative_camera_xyz(x, y, z, x_units=units, y_units=units, z_units=units)

    def set_undistorted_fpa_image_plane_points(self,
                                               undistorted_x_grid,  # type: ndarray
                                               undistorted_y_grid,  # type: ndarray
                                               ):
        self._undistorted_x_grid = undistorted_x_grid
        self._undistorted_y_grid = undistorted_y_grid

    def create_copy(self):
        copy = self
        return copy
