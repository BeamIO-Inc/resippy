from __future__ import division

import numpy as np
from numpy import ndarray
from resippy.image_objects.earth_overhead.abstract_earth_overhead_image import AbstractEarthOverheadImage
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.line_scanner_point_calc import LineScannerPointCalc


class LinescannerImage(AbstractEarthOverheadImage):
    """Concrete implementations should initialize an image for reading/writing
    and should also set the image's metadata object and point calculator object"""

    def __init__(self):
        super(LinescannerImage, self).__init__()
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
        image_data = self.get_image_band(band_number)
        if image_data is not None:
            return image_data
        else:
            if self.read_with_gdal:
                band = self._dset.GetRasterBand(band_number + 1)
            return band.ReadAsArray()

    @property
    def pointcalc(self):  # type: (...) -> LineScannerPointCalc
        return self._point_calc