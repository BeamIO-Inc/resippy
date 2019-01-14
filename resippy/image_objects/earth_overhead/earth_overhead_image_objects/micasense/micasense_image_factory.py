from __future__ import division

from resippy.image_objects.earth_overhead.earth_overhead_image_objects.micasense.\
    micasense_image import MicasenseImage
from resippy.image_objects.earth_overhead.earth_overhead_image_objects.micasense.\
    micasense_metadata import MicasenseMetadata
from resippy.image_objects.earth_overhead.earth_overhead_image_objects.micasense.\
    micasense_pix4d_point_calc import MicasensePix4dPointCalc

import os
from uuid import uuid4
from PIL import Image
import numpy as np


class MicasenseImageFactory:
    @staticmethod
    def from_image_number_and_pix4d(image_fname_dict,  # type: dict
                                    pix4d_master_params_dict  # type: dict
                                    ):  # type: (...) -> MicasenseImage
        micasense_image = MicasenseImage().init_from_image_number_and_pix4d(image_fname_dict, pix4d_master_params_dict)
        return micasense_image

    @staticmethod
    def from_numpy_and_pix4d(image_data,  # type: np.ndarray
                             metadata,  # type: MicasenseMetadata
                             point_calculator,  # type: MicasensePix4dPointCalc
                             working_dir=None,  # type: str
                             ):  # type: (...) -> MicasenseImage

        def write_tmp_band(band_data, working_dir=None):
            uname = os.path.basename(os.getenv("HOME"))

            if working_dir is None:
                output_dir = os.path.join("/tmp", uname)
                os.makedirs(output_dir, exist_ok=True)
            else:
                output_dir = working_dir

            output_fname = os.path.join(output_dir, str(uuid4()) + ".tif")
            out_img = Image.fromarray(band_data)
            out_img.save(output_fname)

            return output_fname

        micasense_image = MicasenseImage()

        ny_nx_nbands = np.shape(image_data)
        nbands = ny_nx_nbands[2]
        for band in range(nbands):
            fn = write_tmp_band(np.squeeze(image_data[:,:,band]), working_dir)
            micasense_image.band_fnames.append(fn)

        micasense_image.set_metadata(metadata)
        micasense_image.set_point_calculator(point_calculator)

        return micasense_image
