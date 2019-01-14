from __future__ import division

import abc
from numpy import ndarray
from resippy.image_objects.abstract_image_metadata import AbstractImageMetadata
from six import add_metaclass


@add_metaclass(abc.ABCMeta)
class AbstractImage():
    """Concrete implementations should initialize an image for reading/writing
    and should also set the image's metadata object and point calculator
    object"""

    def __init__(self):
        self._image_data = None
        self._metadata = None

    @abc.abstractmethod
    def read_all_image_data_from_disk(self):  # type: (...) -> ndarray
        """Return an image of dimensions nbands x Y x X, where
        nbands is the total number of spectral bands"""

    @abc.abstractmethod
    def read_band_from_disk(self,
                            band_number     # type: int
                            ):              # type: (...) -> ndarray
        """Return an image of dimensions Y x X"""

    def set_image_data(self,
                       image_data   # type: ndarray
                       ):           # type: (...) -> None
        self._image_data = image_data

    def get_image_data(self):   # type: (...) -> ndarray
            return self._image_data

    def set_metadata(self,
                     metadata   # type: AbstractImageMetadata
                     ):         # type: (...) -> None
        self._metadata = metadata

    def get_metadata(self):     # type: (...) -> AbstractImageMetadata
            return self._metadata

    def get_image_band(self,
                       band_num     # type: int
                       ):           # type: (...) -> ndarray
        return self._image_data[:, :, band_num]

    def set_image_band(self,
                       image_data,  # type: ndarray
                       band_num     # type: int
                       ):           # type: (...) -> None
        self._image_data[:, :, band_num] = image_data
