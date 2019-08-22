from __future__ import division

import gdal
import numpy as np
from numpy import ndarray

from resippy.image_objects.earth_overhead.abstract_earth_overhead_image import AbstractEarthOverheadImage
from resippy.image_objects.earth_overhead.digital_globe.view_ready_stereo.view_ready_stereo_metadata \
    import ViewReadyStereoMetadata
from resippy.image_objects.earth_overhead.digital_globe.view_ready_stereo.view_ready_stereo_point_calc \
    import ViewReadyStereoPointCalc


class ViewReadyStereoImage(AbstractEarthOverheadImage):
    """Concrete implementations should initialize an image for reading/writing
    and should also set the image's metadata object and point calculator object"""

    def __init__(self):
        super(ViewReadyStereoImage, self).__init__()
        self._dset = None

    @classmethod
    def init_from_file(cls,
                       fname  # type: str
                       ):
        vrs_image = cls()
        vrs_image.set_dset(gdal.Open(fname, gdal.GA_ReadOnly))
        metadata = ViewReadyStereoMetadata._from_file(fname)
        point_calc = ViewReadyStereoPointCalc.init_from_file(fname)
        vrs_image.set_metadata(metadata)
        vrs_image.set_point_calculator(point_calc)
        return vrs_image

    def get_metadata(self):  # type: (...) -> GeotiffMetadata
        return super(ViewReadyStereoImage, self).get_metadata()

    def get_point_calculator(self):  # type: (...) -> GeotiffPointCalc
        return super(ViewReadyStereoImage, self).get_point_calculator()

    def set_dset(self,
                 dset  # type: gdal.Dataset
                 ):   # type: (...) -> None
        self._dset = dset

    def get_dset(self):  # type: (...) -> gdal.Dataset
        return self._dset

    def read_all_image_data_from_disk(self):  # type: (...) -> ndarray
        numpy_arr = []
        for bandnum in range(self.get_metadata().get_n_bands()):
            band = self._dset.GetRasterBand(bandnum + 1)
            numpy_arr.append(band.ReadAsArray())
        return np.stack(numpy_arr, axis=2)

    def read_band_from_disk(self,
                            band_number  # type: int
                            ):  # type: (...) -> ndarray
        band = self._dset.GetRasterBand(band_number + 1)
        return band.ReadAsArray()
