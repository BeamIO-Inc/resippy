from __future__ import division

import numpy as np
from numpy import ndarray
from resippy.image_objects.earth_overhead.abstract_earth_overhead_image import AbstractEarthOverheadImage


class PhysicalCameraImage(AbstractEarthOverheadImage):
    """Concrete implementations should initialize an image for reading/writing
    and should also set the image's metadata object and point calculator object"""

    def __init__(self):
        super(PhysicalCameraImage, self).__init__()
        self._dset = None
        self.read_with_gdal = True

    def read_all_image_data_from_disk(self):  # type: (...) -> ndarray
        numpy_arr = []
        if self.read_with_gdal:
            for bandnum in range(self.get_metadata().get_n_bands()):
                band = self._dset.GetRasterBand(bandnum + 1)
                numpy_arr.append(band.ReadAsArray())
        return np.stack(numpy_arr, axis=2)

    def read_band_from_disk(self,
                            band_number     # type: int
                            ):              # type: (...) -> ndarray
        if self.read_with_gdal:
            band = self._dset.GetRasterBand(band_number + 1)
        return band.ReadAsArray()

    def set_gdal_dset(self,
                      dset  # type: gdal.Dataset
                      ):   # type: (...) -> None
        self._dset = dset

    def get_gdal_dset(self):  # type: (...) -> gdal.Dataset
        return self._dset

    def close_gdal_image(self):  # type: (...) -> None
        self._dset = None

    def __del__(self):
        self._dset = None
