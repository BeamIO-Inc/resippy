import gdal
import os

from resippy.image_objects.earth_overhead.tamarisk.tamarisk_image import TamariskImage
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.opencv_point_calc import OpenCVPointCalc
from resippy.image_objects.earth_overhead.tamarisk.tamarisk_metadata import TamariskMetadata


class TamariskImageFactory:

    @staticmethod
    def from_envi_frame_and_opencv_model(envi_fnames,     # type: dict
                                         opencv_params    # type: dict
                                         ):               # type: (...) -> TamariskImage
        image = TamariskImage()
        image.set_dset(gdal.Open(envi_fnames['image'], gdal.GA_ReadOnly))

        metadata = TamariskMetadata.init_from_header(envi_fnames['header'])
        image.set_metadata(metadata)

        point_calc = OpenCVPointCalc.init_from_params(opencv_params)
        image.set_point_calculator(point_calc)

        return image
