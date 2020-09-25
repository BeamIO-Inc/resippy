from numpy import ndarray

from resippy.utils.units import ureg
from resippy.utils import proj_utils

from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.abstract_earth_overhead_point_calc import AbstractEarthOverheadPointCalc
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.fpa_distortion_mapped_point_calc import FpaDistortionMappedPointCalc


class LineScannerPointCalc(AbstractEarthOverheadPointCalc):
    def __init__(self):
        self._undistorted_x_grid = None     # type: ndarray
        self._undistorted_y_grid = None     # type: ndarray
        self._focal_length_meters = None        # type: ndarray

        self._rolls_radians = []        # type: [float]
        self._pitches_radians = []      # type: [float]
        self._yaws_radians = []         # type: [float]
        self.n_cross_track_pixels = None       # type: int

    def set_camera_model(self,
                         undistorted_image_plane_x_grid,  # type: ndarray
                         undistorted_image_plane_y_grid,  # type: ndarray
                         focal_length,          # type: float
                         focal_length_units='mm'  # type: str
                         ):
        self._undistorted_x_grid = undistorted_image_plane_x_grid
        self._undistorted_y_grid = undistorted_image_plane_y_grid
        self._focal_length_meters = (focal_length * ureg.parse_units(focal_length_units)).to('meters').magnitude
        self.n_cross_track_pixels = undistorted_image_plane_x_grid.shape[0]

    def _lon_lat_alt_to_pixel_x_y_native(self, lons, lats, alts, band=None):
        pass

    def _pixel_x_y_alt_to_lon_lat_native(self, pixel_xs, pixel_ys, alts=None, band=None):
        pass

    def set_xyz_with_wgs84_coords(self, lons, lats, alts, alt_units='meters'):
        native_proj = proj_utils.decimal_degrees_to_local_utm_proj(lons[0],
                                                                   lats[0])
        self.set_projection(native_proj)
        for lon, lat in zip(lons, lats):
            stop = 1

    def set_roll_pitch_yaws(self, rolls, pitches, yaws, units='radians'):
        rolls_radians = (rolls * ureg.parse_units(units)).to('radians')
        pitches_radians = (pitches * ureg.parse_units(units)).to('radians')
        yaws_radians = (yaws * ureg.parse_units(units)).to('radians')
        self._rolls_radians = rolls_radians
        self._pitches_radians = pitches_radians
        self._yaws_radians = yaws_radians

