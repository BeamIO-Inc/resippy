from __future__ import division

from resippy.image_objects.abstract_image import AbstractImage
from resippy.image_objects.envi.envi_metadata import EnviMetadata

import gdal
from numpy import ndarray
import numpy as np
import os


class EnviImage(AbstractImage):
    def __init__(self):
        super(EnviImage, self).__init__()
        self._dset = None

    @classmethod
    def init_from_file(cls,
                       image_file_path,         # type: str
                       header_file_path=None    # type: str
                       ):  # type: (...) -> EnviImage
        envi_image = cls()
        envi_image.set_dset(gdal.Open(image_file_path, gdal.GA_ReadOnly))

        # try to guess the header file name if one is not provided
        if header_file_path is None:
            header_guesses = [image_file_path + ".hdr",
                              image_file_path[0:-4] + ".hdr"]
            for header_guess in header_guesses:
                if os.path.exists(header_guess):
                    header_file_path = header_guess

        metadata = EnviMetadata.init_from_header(header_file_path)
        envi_image.set_metadata(metadata)
        return envi_image

    def read_band_from_disk(self,
                            band_number  # type: int
                            ):  # type: (...) -> ndarray
        band = self._dset.GetRasterBand(band_number + 1)
        return band.ReadAsArray()

    def read_all_image_data_from_disk(self):  # type: (...) -> ndarray
        numpy_arr = []
        for bandnum in range(self.get_metadata().get_n_bands()):
            band = self._dset.GetRasterBand(bandnum + 1)
            numpy_arr.append(band.ReadAsArray())
        return np.stack(numpy_arr, axis=2)

    def set_dset(self,
                 dset  # type: gdal.Dataset
                 ):   # type: (...) -> None
        self._dset = dset
