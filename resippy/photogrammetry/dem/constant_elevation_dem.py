from __future__ import division

from resippy.photogrammetry.dem.abstract_dem import AbstractDem
import numpy as np
import resippy.photogrammetry.crs_defs as crs_defs


class ConstantElevationDem(AbstractDem):
    def __init__(self, elevation=0):
        self.elevation = elevation

    def get_elevations(self,
                       world_x,  # type: ndarray
                       world_y,  # type: ndarray
                       world_proj=crs_defs.PROJ_4326  # type: Proj
                       ):  # type: (...) -> ndarray
        return np.zeros(np.shape(world_x)) + self.elevation

    def get_highest_alt(self):  # type: (...) -> float
        return self.elevation

    def get_lowest_alt(self):  # type: (...) -> float
        return self.elevation

    # TODO MAYBE USE PROJ4 HERE?
    def convert_reference(self,
                          dst_fname,  # type: str
                          dst_epsg_code  # type: int
                          ):  # type: (...) -> None
        return None
