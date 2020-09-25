from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.abstract_earth_overhead_point_calc import AbstractEarthOverheadPointCalc
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.fpa_distortion_mapped_point_calc import FpaDistortionMappedPointCalc


class LineScanner(AbstractEarthOverheadPointCalc):
    def __init__(self):
        pass

    def _lon_lat_alt_to_pixel_x_y_native(self, lons, lats, alts, band=None):
        pass

    def _pixel_x_y_alt_to_lon_lat_native(self, pixel_xs, pixel_ys, alts=None, band=None):
        pass

