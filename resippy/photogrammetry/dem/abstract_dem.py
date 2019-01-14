from __future__ import division

import abc
from pyproj import Proj
from numpy import ndarray
from six import add_metaclass


@add_metaclass(abc.ABCMeta)
class AbstractDem():
    """Concrete implementations should initialize an image for reading/writing
    and should also set the image's metadata object and point calculator object"""

    def __init__(self, **kwargs):
        self.proj = None
        self.crs = None
        # EPSG Code see: https://vdatum.noaa.gov/docs/datums.html#verticaldatum
        # Codes can be found at: https://epsg.io/
        # Example: https://epsg.io/5773 is the geoid commonly used by pix4d
        self.reference = None

    def set_projection(self,
                       projection  # type: Proj
                       ):  # type: (...) -> None
        self.proj = projection

    def get_projection(self):  # type: (...) -> Proj
        return self.proj

    def set_reference(self,
                      epsg_code  # type: int
                      ):  # type: (...) -> None
        self.reference = epsg_code

    def get_reference(self):  # type: (...) -> int
        return self.reference

    @abc.abstractmethod
    def convert_reference(self,
                          dst_fname,  # type: str
                          dst_epsg_code  # type: int
                          ):  # type: (...) -> None
        pass

    @abc.abstractmethod
    def get_elevations(self,
                       world_x,  # type: ndarray
                       world_y,  # type: ndarray
                       world_proj=None  # type: Proj
                       ):  # type: (...) -> ndarray
        pass

    @abc.abstractmethod
    def get_highest_alt(self):  # type: (...) -> float
        pass

    @abc.abstractmethod
    def get_lowest_alt(self):  # type: (...) -> float
        pass
