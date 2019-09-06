import numpy as np
from pyproj import Proj

from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.abstract_earth_overhead_point_calc \
    import AbstractEarthOverheadPointCalc


class EarthOverheadSensorModel(AbstractEarthOverheadPointCalc):

    def __init__(self):     # type: (...) -> EarthOverheadSensorModel
        super().__init__()
        self._point_calcs = []

    def _lon_lat_alt_to_pixel_x_y_native(self,
                                         lons,      # type: np.ndarray
                                         lats,      # type: np.ndarray
                                         alts,      # type: np.ndarray
                                         band=None  # type: int
                                         ):         # type: (...) -> (np.ndarray, np.ndarray)
        if band < len(self._point_calcs):
            return self._point_calcs[band]._lon_lat_alt_to_pixel_x_y_native(lons, lats, alts, band)

        # TODO: error for band out of range

    def _pixel_x_y_alt_to_lon_lat_native(self,
                                         pixel_xs,      # type: np.ndarray
                                         pixel_ys,      # type: np.ndarray
                                         alts=None,     # type: np.ndarray
                                         band=None      # type: int
                                         ):             # type: (...) -> (np.ndarray, np.ndarray)
        if band < len(self._point_calcs):
            return self._point_calcs[band]._pixel_x_y_alt_to_lon_lat_native(pixel_xs, pixel_ys, alts, band)

        # TODO: error for band out of range

    def set_projection(self,
                       projection   # type: Proj
                       ):           # type: (...) -> None
        super().set_projection(projection)
        [point_calc.set_projection(projection) for point_calc in self._point_calcs]

    def set_distortion_model(self,
                             distortion_model   # type: str
                             ):                 # type: (...) -> None
        [point_calc.set_distortion_model(distortion_model) for point_calc in self._point_calcs]

    def set_point_calcs(self,
                        point_calcs     # type: [AbstractEarthOverheadPointCalc]
                        ):              # type: (...) -> None
        self._point_calcs = point_calcs

    def get_point_calcs(self
                        ):      # type: (...) -> [AbstractEarthOverheadPointCalc]
        return self._point_calcs

    def add_point_calc(self,
                       point_calc   # type: AbstractEarthOverheadPointCalc
                       ):           # type: (...) -> None
        self._point_calcs.append(point_calc)

    def update_point_calc(self,
                          index,        # type: int
                          point_calc    # type: AbstractEarthOverheadPointCalc
                          ):            # type: (...) -> None
        self._point_calcs[index] = point_calc

    def get_point_calc(self,
                       index    # type: int
                       ):       # type: (...) -> AbstractEarthOverheadPointCalc
        return self._point_calcs[index]

    def get_num_point_calcs(self
                            ):      # type: (...) -> int
        return len(self._point_calcs)
