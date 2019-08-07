from __future__ import division

from resippy.image_objects.earth_overhead.micasense.micasense_metadata import MicasenseMetadata
from resippy.image_objects.earth_overhead.abstract_earth_overhead_image import AbstractEarthOverheadImage

from numpy import ndarray
import numpy as np
from imageio import imread
import os


class MicasenseImage(AbstractEarthOverheadImage):
    def __init__(self):
        super(MicasenseImage, self).__init__()
        self.band_fnames = []

    def read_all_image_data_from_disk(self):  # type: (...) -> ndarray
        imgs = [imread(fname) for fname in self.band_fnames]
        all_image_data = np.dstack(imgs)

        return all_image_data

    def read_band_from_disk(self,
                            band_number  # type: int
                            ):  # type: (...) -> ndarray
        return imread(self.band_fnames[band_number])
