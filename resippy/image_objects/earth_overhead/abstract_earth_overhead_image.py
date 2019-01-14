from __future__ import division

from numpy import ndarray
from resippy.image_objects.abstract_image import AbstractImage
from resippy.image_objects.abstract_image_metadata import AbstractImageMetadata
from resippy.image_objects.earth_overhead.abstract_earth_overhead_point_calc import AbstractEarthOverheadPointCalc
import abc


class AbstractEarthOverheadImage(AbstractImage):
    """Concrete implementations should initialize an image for reading/writing
    and should also set the image's metadata object and point
    calculator object"""

    def __init__(self):
        super(AbstractEarthOverheadImage, self).__init__()
        self._point_calc = AbstractEarthOverheadPointCalc
        self._metadata = AbstractImageMetadata

    def get_point_calculator(self):  # type: (...) -> AbstractEarthOverheadPointCalc
        return self._point_calc

    @abc.abstractmethod
    def read_all_image_data_from_disk(self):  # type: (...) -> ndarray
        pass

    @abc.abstractmethod
    def read_band_from_disk(self,
                            band_number  # type: int
                            ):  # type: (...) -> ndarray
        pass

    def set_point_calculator(self,
                             point_calculator  # type: AbstractEarthOverheadPointCalc
                             ):  # type: (...) -> None
        self._point_calc = point_calculator
