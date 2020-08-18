import gdal

from resippy.image_objects.earth_overhead.mako.mako_image import MakoImage
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.opencv_point_calc import OpenCVPointCalc
from resippy.image_objects.earth_overhead.mako.mako_metadata import MakoMetadata


class MakoImageFactory:

    @staticmethod
    def from_envi_frame_and_opencv_model(envi_fnames,     # type: dict
                                         opencv_params    # type: dict
                                         ):               # type: (...) -> MakoImage
        image = MakoImage()
        image.set_dset(gdal.Open(envi_fnames['image'], gdal.GA_ReadOnly))

        metadata = MakoMetadata.init_from_header(envi_fnames['header'])
        image.set_metadata(metadata)

        point_calc = OpenCVPointCalc.init_from_params(opencv_params)
        image.set_point_calculator(point_calc)

        return image
