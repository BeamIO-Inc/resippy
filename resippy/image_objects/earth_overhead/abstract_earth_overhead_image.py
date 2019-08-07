from __future__ import division

from numpy import ndarray
from resippy.image_objects.abstract_image import AbstractImage
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.earth_overhead_sensor_model import EarthOverheadSensorModel
from resippy.image_objects.abstract_image_metadata import AbstractImageMetadata
import abc


class AbstractEarthOverheadImage(AbstractImage):
    """
    This is the AbstractEarthOverheadImage Object.  It is an abstraction for a specific type of image object, which
    is an image taken of the earth from overhead.  This type of image object has a point calculator object, which
    is used to map from world coordinates (longitude, latitude) to pixel coordinates (x, y) and vice versa.
    Concrete implementations should initialize an image for reading/writing
    and should also set the image's metadata object and point
    calculator object"""

    def __init__(self):
        super(AbstractEarthOverheadImage, self).__init__()
        self._sensor_model = EarthOverheadSensorModel
        self._metadata = AbstractImageMetadata

    def get_sensor_model(self):  # type: (...) -> EarthOverheadSensorModel
        """
        Returns the image object's sensor model
        :return: a sensor model.  The types of point calculator in the sensor modelwill be determined by the
        corresponding image object.  All point calculators are concrete implementations of the
        AbstractEarthOverheadPointCalc class and should all be able to be used the same way, regardless of the specific
        type of point calculator.
        """
        return self._sensor_model

    @abc.abstractmethod
    def read_all_image_data_from_disk(self):  # type: (...) -> ndarray
        """
        Reads all of the image object's data from disk, if the image has data stored on disk.
        :return: all image data, as a numpy ndarray.  The dimensions should be (ny, nx, nbands)
        """
        pass

    @abc.abstractmethod
    def read_band_from_disk(self,
                            band_number  # type: int
                            ):  # type: (...) -> ndarray
        """
        Reads a single image band from disk
        :param band_number: band number to read from disk
        :return: numpy ndarray of dimensions (ny, nx) pixels.
        """
        pass

    def set_sensor_model(self,
                         sensor_model  # type: EarthOverheadSensorModel
                         ):  # type: (...) -> None
        """
        Sets the point calculator for the image object.  This should only be used when creating a new type of
        EarthOverheadImage object.
        :param sensor_model: The EarthOverheadSensorModel containing point calculators for each focal plane.
        :return: None
        """
        self._sensor_model = sensor_model
