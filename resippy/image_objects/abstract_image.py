from __future__ import division

import abc
from numpy import ndarray
from resippy.image_objects.abstract_image_metadata import AbstractImageMetadata
from six import add_metaclass


@add_metaclass(abc.ABCMeta)
class AbstractImage():
    """
    This is the AbstractImage class.  Instantiating this class provides functionality to read data from disk, if image
    data for the class comes from data on disk.  There are routines to manipulate image data and metadata.
    Each concrete implementation of this abstract class should be bundled with a metadata object, and an ImageFactory
    Object that is used to initialize and create image objects, either from disk or from user parameters.
    """

    def __init__(self):
        """
        This is the default initialization method.  It sets protected class variables for the image data and metadata
        and initializes them to 'None'.  Generally initializing an image object with this init method will not be
        used in practice.  Instead, the ImageFactory class is provided to generate image objects and should be used
        to instantiate Image Object instances whenever possible.
        """
        self._image_data = None
        self._metadata = None

    @abc.abstractmethod
    def read_all_image_data_from_disk(self):  # type: (...) -> ndarray
        """
        Reads all image data from disk.  Care should be used with this method when dealing with large data files.
        The format of imagery read from disk should be (ny, nx, nbands).  If imagery is a single band it should
        still be in this format, specifically (ny, nx, 1).  All image data is assumed to have 3 dimensions by default
        image 'bands' have dimensions of (ny, nx)
        :return:  ndarray containing image data of dimensions (ny, nx, nbands)
        """

    @abc.abstractmethod
    def read_band_from_disk(self,
                            band_number     # type: int
                            ):              # type: (...) -> ndarray
        """
        Reads a specific image band from disk.
        :param band_number: band number to be read in.  This is zero-based (first channel => band_number=0)
        :return: ndarray containing image data of dimensions (ny, nx)
        """

    def set_image_data(self,
                       image_data   # type: ndarray
                       ):           # type: (...) -> None
        """
        Sets the image object's image data using the _image_data class variable
        :param image_data: ndarray of dimensions (ny, nx, nbands).  This will overwrite the image object's image data
        :return: None
        """
        self._image_data = image_data

    def get_image_data(self):   # type: (...) -> ndarray
        """
        gets the image data that has been loaded into the class's _image_data class variable
        :return: ndarray of dimensions (ny, nx, nbands)
        """
        return self._image_data

    def set_metadata(self,
                     metadata   # type: AbstractImageMetadata
                     ):         # type: (...) -> None
        """
        Sets the image object's metadata, and stores it in the image object's _metadata class variable.
        :param metadata: ImageMetadata class.
        :return: None
        """
        self._metadata = metadata

    def get_metadata(self):     # type: (...) -> AbstractImageMetadata
        """
        gets the image object's metadata
        :return: ImageMetadata class.  This will be a concrete implementation of AbstractImageMetadata, and will
        match the ImageMetadata class associated with the particular concrete implementation of the Abstract Image.
        For example GeotiffImageObject will have a GeotiffImageMetadata.
        """
        return self._metadata

    def get_image_band(self,
                       band_num     # type: int
                       ):           # type: (...) -> ndarray
        """
        Gets a particular image band of the image object's _image_data class variable.  band_num is zero-based.
        :param band_num: band number to get
        :return: image band of dimensions (ny, nx)
        """
        return self._image_data[:, :, band_num]

    def set_image_band(self,
                       image_data,  # type: ndarray
                       band_num     # type: int
                       ):           # type: (...) -> None
        """
        Sets or overwrites a particular image band of the image object's _image_data class variable
        :param image_data: image band of dimensions (ny, nx)
        :param band_num: band number to overwrite
        :return: None
        """
        self._image_data[:, :, band_num] = image_data
