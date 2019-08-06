from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.pinhole_camera_point_calc \
    import PinholeCameraPointCalc

import numpy as np


class PhysicalModelPointCalc(PinholeCameraPointCalc):

    def __init__(self):
        super(PhysicalModelPointCalc, self).__init__()

        # TODO: initialize distortion model
        # TODO: initialize exterior orientation

    @classmethod
    def init_from_coeffs(cls
                         ):     # type: (...) -> PhysicalModelPointCalc
        # TODO
        pass

    @classmethod
    def init_from_file(cls,
                       filename     # type: str
                       ):           # type: (...) -> PhysicalModelPointCalc
        # TODO
        pass

    def _lon_lat_alt_to_pixel_x_y_native(self,
                                         lons,          # type: np.ndarray
                                         lats,          # type: np.ndarray
                                         alts=None,     # type: np.ndarray
                                         band=None      # type: int
                                         ):             # type: (...) -> (np.ndarray, np.ndarray)
        # TODO
        pass
