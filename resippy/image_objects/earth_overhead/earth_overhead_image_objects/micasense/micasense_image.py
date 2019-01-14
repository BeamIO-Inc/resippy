from __future__ import division

from resippy.image_objects.earth_overhead.earth_overhead_image_objects.micasense.\
    micasense_metadata import MicasenseMetadata
from resippy.image_objects.earth_overhead.earth_overhead_image_objects.micasense.\
    micasense_pix4d_point_calc import MicasensePix4dPointCalc
from resippy.image_objects.earth_overhead.abstract_earth_overhead_image import AbstractEarthOverheadImage

from numpy import ndarray
import numpy as np
import scipy.misc as misc


class MicasenseImage(AbstractEarthOverheadImage):
    def __init__(self):
        super(MicasenseImage, self).__init__()
        self.band_fnames = []

    def read_all_image_data_from_disk(self):  # type: (...) -> ndarray
        imgs = [misc.imread(fname) for fname in self.band_fnames]
        all_image_data = np.dstack(imgs)

        return all_image_data

    def read_band_from_disk(self,
                            band_number  # type: int
                            ):  # type: (...) -> ndarray
        return misc.imread(self.band_fnames[band_number])

    @classmethod
    def init_from_image_number_and_pix4d(cls,
                                         band_fname_dict,  # type: dict
                                         pix4d_master_params_dict  # type: dict
                                         ):  # type: (...) -> MicasenseImage
        micasense_image = MicasenseImage()

        point_calc = MicasensePix4dPointCalc().init_from_params(band_fname_dict, pix4d_master_params_dict)

        micasense_image.band_fnames.append(band_fname_dict['band1'])
        micasense_image.band_fnames.append(band_fname_dict['band2'])
        micasense_image.band_fnames.append(band_fname_dict['band3'])
        micasense_image.band_fnames.append(band_fname_dict['band4'])
        micasense_image.band_fnames.append(band_fname_dict['band5'])

        metadata = MicasenseMetadata()
        micasense_image.set_metadata(metadata)
        micasense_image.set_point_calculator(point_calc)

        return micasense_image
