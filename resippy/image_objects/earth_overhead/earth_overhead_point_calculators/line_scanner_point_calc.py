import numbers

from numpy import ndarray
import numpy
from pyproj import transform

from resippy.utils.units import ureg
from resippy.utils import proj_utils
from resippy.photogrammetry import crs_defs

from resippy.utils.image_utils import image_utils
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.abstract_earth_overhead_point_calc import AbstractEarthOverheadPointCalc
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.fpa_distortion_mapped_point_calc import FpaDistortionMappedPointCalc


class LineScannerPointCalc(AbstractEarthOverheadPointCalc):
    def __init__(self):
        self._distorted_x_grid = None     # type: ndarray
        self._distorted_y_grid = None     # type: ndarray
        self._focal_length = None        # type: ndarray
        self._focal_length_units = None     # type: ndarray

        self.local_lons = []
        self.local_lats = []
        self.alts = []
        self.alt_units = None

        self._rolls = []        # type: [float]
        self._pitches = []      # type: [float]
        self._yaws = []         # type: [float]
        self.roll_pitch_yaw_units = None
        self._roll_pitch_yaw_order = None

        self._mount_offset_x_meters = 0
        self._mount_offset_y_meters = 0
        self._mount_offset_z_meters = 0
        self._mount_offset_units = None

        self._boresight_roll = 0
        self._boresight_pitch = 0
        self._boresight_yaw = 0
        self._boresight_units = None
        self._boresight_rpy_order = None

        self.set_boresight_offsets(0, 0, 0, units='radians')

        self._npix_x = None
        self._npix_y = None

    def set_focal_length(self,
                         focal_length,  # type: float
                         focal_length_units,  # type: str
                         ):
        self._focal_length = focal_length
        self._focal_length_units = focal_length_units

    def set_camera_model_w_distorted_image_plane_grid(self,
                                                      distorted_image_plane_x_grid,  # type: ndarray
                                                      distorted_image_plane_y_grid,  # type: ndarray
                                                      focal_length,  # type: float
                                                      focal_length_units  # type: str
                                                      ):
        self._distorted_x_grid = distorted_image_plane_x_grid
        self._distorted_y_grid = distorted_image_plane_y_grid
        self._focal_length = focal_length
        self._focal_length_units = focal_length_units
        self._npix_y = len(distorted_image_plane_x_grid)

    def set_camera_model_w_ideal_pinhole_model(self,
                                               n_pixels,
                                               focal_length,
                                               pixel_pitch,
                                               focal_length_units,
                                               pixel_pitch_units):
        self._npix_y = n_pixels
        self._focal_length = focal_length
        self._focal_length_units = focal_length_units
        pp_meters = (pixel_pitch * ureg(pixel_pitch_units)).to('meters').magnitude
        image_plane_grid = image_utils.create_image_plane_grid(1, n_pixels, pp_meters, pp_meters)
        self._distorted_x_grid = image_plane_grid[0]
        self._distorted_y_grid = image_plane_grid[1]

    def _lon_lat_alt_to_pixel_x_y_native(self, lons, lats, alts, band=None):
        pass

    def _pixel_x_y_alt_to_lon_lat_native(self, pixel_xs, pixel_ys, alts=None, band=None):
        input_type = "array"
        if isinstance(pixel_xs, numbers.Number):
            input_type = "number"
            pixel_xs = numpy.asarray([pixel_xs])
            pixel_ys = numpy.asarray([pixel_ys])
        lons_native = numpy.zeros_like(pixel_xs)
        lats_native = numpy.zeros_like(pixel_xs)
        lines_to_project = set(pixel_xs.ravel())
        # TODO: at some point we'll need to address non integer x values
        for line in lines_to_project:
            single_line_point_calc = FpaDistortionMappedPointCalc()
            single_line_point_calc.set_distorted_fpa_image_plane_points(self._distorted_x_grid,
                                                                        self._distorted_y_grid)
            single_line_point_calc.set_focal_length(self._focal_length, units=self._focal_length_units)
            single_line_point_calc.set_projection(self.get_projection())
            single_line_point_calc.set_xyz_using_local_coords(self.local_lons[line],
                                                              self.local_lats[line],
                                                              self.alts[line],
                                                              alt_units=self.alt_units)
            single_line_point_calc.set_roll_pitch_yaw(self._rolls[line],
                                                      self._pitches[line],
                                                      self._yaws[line],
                                                      units=self.roll_pitch_yaw_units,
                                                      order='rpy')
            single_line_point_calc.set_mounting_position_on_fixture(self._mount_offset_x_meters,
                                                                    self._mount_offset_y_meters,
                                                                    self._mount_offset_z_meters,
                                                                    units='meters')
            single_line_point_calc.set_boresight_roll_pitch_yaw_offsets(self._boresight_roll,
                                                                        self._boresight_pitch,
                                                                        self._boresight_yaw,
                                                                        units=self._boresight_units,
                                                                        order=self._boresight_rpy_order)
            indices = numpy.where(pixel_xs == line)

            if isinstance(alts, numbers.Number):
                alts = numpy.zeros_like(pixel_xs) + alts
            if len(indices) == 2:
                line_pixels_to_project_y = pixel_ys[indices[0], indices[1]]
                line_alts = alts[indices[0], indices[1]]
            else:
                line_pixels_to_project_y = pixel_ys[indices]
                line_alts = alts[indices]
            x_pixels_to_project = numpy.zeros_like(line_pixels_to_project_y)
            line_lons, line_lats = single_line_point_calc._pixel_x_y_alt_to_lon_lat_native(x_pixels_to_project,
                                                                                           line_pixels_to_project_y,
                                                                                           line_alts)
            lons_native[indices] = line_lons
            lats_native[indices] = line_lats
        if input_type == "number":
            lons_native = lons_native[0]
            lats_native = lats_native[0]
        return lons_native, lats_native

    def set_xyz_with_local_coords(self, lons, lats, alts, alt_units, local_proj):
        self.set_projection(local_proj)
        self.local_lons = lons
        self.local_lats = lats
        self.alts = alts
        self.alt_units = alt_units
        self._npix_x = len(lons)

    def set_xyz_with_wgs84_coords(self, lons, lats, alts, alt_units):
        local_proj = proj_utils.decimal_degrees_to_local_utm_proj(lons[0],
                                                                   lats[0])
        local_lons, local_lats = transform(crs_defs.PROJ_4326,
                                           local_proj,
                                           lons,
                                           lats)
        self.set_xyz_with_local_coords(local_lons, local_lats, alts, alt_units, local_proj)

    def set_roll_pitch_yaws(self,
                            rolls,  # type: ndarray
                            pitches,  # type: ndarray
                            yaws,   # type: ndarray
                            units,  # type: str
                            order='rpy'):
        self._rolls = rolls
        self._pitches = pitches
        self._yaws = yaws
        self.roll_pitch_yaw_units = units
        self._roll_pitch_yaw_order = order

    def set_mounting_position_on_fixture(self, x, y, z, units):
        self._mount_offset_x_meters = (x * ureg.parse_units(units)).to('meters').magnitude
        self._mount_offset_y_meters = (y * ureg.parse_units(units)).to('meters').magnitude
        self._mount_offset_z_meters = (z * ureg.parse_units(units)).to('meters').magnitude

    def set_boresight_offsets(self, roll, pitch, yaw, units, order='rpy'):
        self._boresight_roll = roll
        self._boresight_pitch = pitch
        self._boresight_yaw = yaw
        self._boresight_units = units
        self._boresight_rpy_order = order

    def get_gsd_at_nadir(self,
                     line=None,  # type: int
                     ):
        center_pixel = int(self._npix_y / 2)
        if line is not None:
            alt = self.alts[line]
        else:
            alt = numpy.mean(self.alts)
        pixel_pitch = numpy.abs(self._distorted_y_grid[center_pixel] - self._distorted_y_grid[center_pixel - 1])
        pixel_pitch = pixel_pitch[0] * ureg.parse_units('meters')
        gsd = pixel_pitch * (alt * ureg.parse_units(self.alt_units)) / \
              (self._focal_length * ureg.parse_units(self._focal_length_units))
        return gsd.to('meters').magnitude

    def get_ifov_at_optical_center(self, units="radians"):
        n_pixels = self._distorted_y_grid.shape[0]
        pixel_1 = self._distorted_y_grid[int(n_pixels/2)]
        pixel_2 = self._distorted_y_grid[int(n_pixels/2 + 1)]
        pp = numpy.abs(pixel_2 - pixel_1)
        flen_meters = (self._focal_length * ureg.parse_units(self._focal_length_units)).to('meters').magnitude
        ifov = numpy.arctan(pp / flen_meters)
        if units == "degrees":
            return numpy.rad2deg(ifov)
        else:
            return ifov
