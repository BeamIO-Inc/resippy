import numpy as np

from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.abstract_earth_overhead_point_calc \
    import AbstractEarthOverheadPointCalc


class EarthOverheadSensorModel:

    def __init__(self
                 ):     # type: (...) -> EarthOverheadSensorModel
        self._point_calcs = []

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
