from __future__ import division

from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.pinhole_camera import PinholeCamera
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.abstract_earth_overhead_point_calc import AbstractEarthOverheadPointCalc
from resippy.photogrammetry import crs_defs
import utm
from pyproj import Proj
from pyproj import transform
import numpy as np


class AgisoftQuickPreviewPointCalc(AbstractEarthOverheadPointCalc):

    def __init__(self):
        super(AgisoftQuickPreviewPointCalc, self).__init__()
        self._pinhole_camera=None       # type: PinholeCamera
        self._lon_lat_center_approximate = None
        self._pixel_pitch_meters = None
        self._npix_x = None
        self._npix_y = None
        self._flip_x = False
        self._flip_y = False

    def _pixel_x_y_alt_to_lon_lat_native(self, pixel_xs, pixel_ys, alts=None, band=None):
        pass

    def _lon_lat_alt_to_pixel_x_y_native(self, lons, lats, alts, band=None):
        half_fpa_x_meters = (self._npix_x * self._pixel_pitch_meters)/2.0
        half_fpa_y_meters = (self._npix_y * self._pixel_pitch_meters)/2.0
        fpa_coords_meters_x, fpa_coords_meters_y = self._pinhole_camera._world_to_image_space(lons, lats, alts)
        if self._flip_x:
            fpa_coords_meters_x = -1.0 * fpa_coords_meters_x
        if self._flip_y:
            fpa_coords_meters_y = -1.0 * fpa_coords_meters_y
        fpa_coords_pixels_x = (fpa_coords_meters_x + half_fpa_x_meters) / self._pixel_pitch_meters
        fpa_coords_pixels_y = (fpa_coords_meters_y + half_fpa_y_meters) / self._pixel_pitch_meters
        return fpa_coords_pixels_x, fpa_coords_pixels_y


    @classmethod
    def init_from_params(cls,
                         sensor_lon_deg,
                         sensor_lat_deg,
                         sensor_alt_meters_msl,  # type: float
                         roll_degrees,
                         pitch_degrees,
                         yaw_degrees,
                         focal_length_mm,
                         pixel_pitch_microns,
                         npix_x,
                         npix_y,
                         flip_y=False,
                         flip_x=False,
                         ):                         # type: (...) -> AgisoftQuickPreviewPointCalc
        zone_info = utm.from_latlon(sensor_lat_deg, sensor_lon_deg)
        zone_string = str(zone_info[2]) + zone_info[3]
        native_proj = Proj(proj='utm', ellps='WGS84', zone=zone_string, south=False)
        proj_4326 = crs_defs.PROJ_4326

        omega_radians = np.deg2rad(roll_degrees)
        phi_radians = np.deg2rad(pitch_degrees)
        kappa_radians = np.deg2rad(yaw_degrees)

        sensor_x_local, sensor_y_local = transform(proj_4326, native_proj, sensor_lon_deg, sensor_lat_deg)

        agisoft_quick_preview_point_calc = AgisoftQuickPreviewPointCalc()

        pinhole_camera = PinholeCamera()
        pinhole_camera.init_pinhole_from_coeffs(sensor_x_local,
                                                sensor_y_local,
                                                sensor_alt_meters_msl,
                                                omega_radians,
                                                phi_radians,
                                                kappa_radians,
                                                focal_length_mm)
        agisoft_quick_preview_point_calc._pixel_pitch_meters = pixel_pitch_microns * 1e-6
        agisoft_quick_preview_point_calc.set_projection(native_proj)
        agisoft_quick_preview_point_calc.set_approximate_lon_lat_center(sensor_x_local, sensor_y_local)
        agisoft_quick_preview_point_calc._npix_x = npix_x
        agisoft_quick_preview_point_calc._npix_y = npix_y
        agisoft_quick_preview_point_calc._flip_x = flip_x
        agisoft_quick_preview_point_calc._flip_y = flip_y
        agisoft_quick_preview_point_calc._pinhole_camera = pinhole_camera
        return agisoft_quick_preview_point_calc
