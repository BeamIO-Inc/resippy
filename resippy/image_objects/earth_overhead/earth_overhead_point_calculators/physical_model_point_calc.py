from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.pinhole_camera \
    import PinholeCamera
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.abstract_earth_overhead_point_calc \
    import AbstractEarthOverheadPointCalc

import numpy as np


class PhysicalModelPointCalc(PinholeCamera, AbstractEarthOverheadPointCalc):

    def __init__(self):
        # TODO: initialize intrinsic model parameters
        pass

    @classmethod
    def init_from_params(cls,
                         params     # type: dict
                         ):         # type: (...) -> PhysicalModelPointCalc
        point_calc = cls()

        extrinsic_params = params['extrinsic']
        point_calc.init_pinhole_from_coeffs(extrinsic_params['X'], extrinsic_params['Y'], extrinsic_params['Z'],
                                            extrinsic_params['omega'], extrinsic_params['phi'],
                                            extrinsic_params['kappa'], extrinsic_params['focal_length'])

        intrinsic_params = params['intrinsic']
        # TODO: set intrinsic model values from params

        return point_calc

    def _lon_lat_alt_to_pixel_x_y_native(self,
                                         lons,          # type: np.ndarray
                                         lats,          # type: np.ndarray
                                         alts=None,     # type: np.ndarray
                                         band=None      # type: int
                                         ):             # type: (...) -> (np.ndarray, np.ndarray)
        # TODO
        pass

    def _pixel_x_y_alt_to_lon_lat_native(self,
                                         pixel_xs,      # type: np.ndarray
                                         pixel_ys,      # type: np.ndarray
                                         alts=None,     # type: np.ndarray
                                         band=None      # type: np.ndarray
                                         ):             # type: (...) -> (np.ndarray, np.ndarray)
        # TODO
        pass
