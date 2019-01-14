from __future__ import division

from numpy.core.multiarray import ndarray
from pyproj import Proj

from resippy.image_objects.earth_overhead.abstract_earth_overhead_point_calc import AbstractEarthOverheadPointCalc
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.pix4d_point_calc import Pix4dPointCalc


class MicasensePix4dPointCalc(AbstractEarthOverheadPointCalc):
    def __init__(self):
        super(MicasensePix4dPointCalc, self).__init__()
        self.point_calc_1 = Pix4dPointCalc()
        self.point_calc_2 = Pix4dPointCalc()
        self.point_calc_3 = Pix4dPointCalc()
        self.point_calc_4 = Pix4dPointCalc()
        self.point_calc_5 = Pix4dPointCalc()

        self.point_calc_1.reverse_x_pixels = True
        self.point_calc_2.reverse_x_pixels = True
        self.point_calc_3.reverse_x_pixels = True
        self.point_calc_4.reverse_x_pixels = True
        self.point_calc_5.reverse_x_pixels = True

        self.point_calc_1.reverse_y_pixels = True
        self.point_calc_2.reverse_y_pixels = True
        self.point_calc_3.reverse_y_pixels = True
        self.point_calc_4.reverse_y_pixels = True
        self.point_calc_5.reverse_y_pixels = True

        self._bands_coregistered = False

    @classmethod
    def init_from_params(cls,
                         band_fname_dict,  # type: dict
                         all_params  # type:  dict
                         ):  # type: (...) -> MicasensePix4dPointCalc
        point_calc = MicasensePix4dPointCalc()

        point_calc.point_calc_1 = Pix4dPointCalc.init_from_params(band_fname_dict['band1'], all_params)
        point_calc.point_calc_2 = Pix4dPointCalc.init_from_params(band_fname_dict['band2'], all_params)
        point_calc.point_calc_3 = Pix4dPointCalc.init_from_params(band_fname_dict['band3'], all_params)
        point_calc.point_calc_4 = Pix4dPointCalc.init_from_params(band_fname_dict['band4'], all_params)
        point_calc.point_calc_5 = Pix4dPointCalc.init_from_params(band_fname_dict['band5'], all_params)

        approx_center_lon, approx_center_lat = point_calc.point_calc_1.get_approximate_lon_lat_center()
        projection = point_calc.point_calc_1.get_projection()

        point_calc.set_approximate_lon_lat_center(approx_center_lon, approx_center_lat)
        point_calc.set_projection(projection)

        return point_calc

    def _pixel_x_y_alt_to_lon_lat_native(self,
                                         pixel_xs,  # type: ndarray
                                         pixel_ys,  # type: ndarray
                                         alts=None,  # type: ndarray
                                         band=None  # type: int
                                         ):  # type: (...) -> (ndarray, ndarray)
        super(MicasensePix4dPointCalc, self)

    def set_projection(self,
                       projection  # type: Proj
                       ):  # type: (...) -> None
        super(MicasensePix4dPointCalc, self).set_projection(projection)
        self.point_calc_1.set_projection(projection)
        self.point_calc_2.set_projection(projection)
        self.point_calc_3.set_projection(projection)
        self.point_calc_4.set_projection(projection)
        self.point_calc_5.set_projection(projection)

    def _lon_lat_alt_to_pixel_x_y_native(self,
                                         lons,  # type: ndarray
                                         lats,  # type: ndarray
                                         alts,  # type: ndarray
                                         band=None  # type: int
                                         ):  # type: (...) -> (ndarray, ndarray)
        if band == 0:
            return self.point_calc_1._lon_lat_alt_to_pixel_x_y_native(lons, lats, alts)
        if band == 1:
            return self.point_calc_2._lon_lat_alt_to_pixel_x_y_native(lons, lats, alts)
        if band == 2:
            return self.point_calc_3._lon_lat_alt_to_pixel_x_y_native(lons, lats, alts)
        if band == 3:
            return self.point_calc_4._lon_lat_alt_to_pixel_x_y_native(lons, lats, alts)
        if band == 4:
            return self.point_calc_5._lon_lat_alt_to_pixel_x_y_native(lons, lats, alts)

    def set_distortion_model(self,
                             distortion_model  # type: str
                             ):  # type: (...) -> None
        self.point_calc_1.distortion_model = distortion_model
        self.point_calc_2.distortion_model = distortion_model
        self.point_calc_3.distortion_model = distortion_model
        self.point_calc_4.distortion_model = distortion_model
        self.point_calc_5.distortion_model = distortion_model
