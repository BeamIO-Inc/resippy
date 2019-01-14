from __future__ import division

from resippy.image_objects.earth_overhead.earth_overhead_image_objects.geotiff.geotiff_image import GeotiffImage

from numpy import ndarray
from pyproj import Proj


class GeotiffImageFactory:
    @staticmethod
    def from_file(fname  # type:  str
                  ):  # type: (...) -> GeotiffImage
        geotiff_image = GeotiffImage.init_from_file(fname)
        return geotiff_image

    @staticmethod
    def from_numpy_array(image_data,  # type: ndarray
                         geo_t,  # type: list
                         projection,  # type: Proj
                         nodata_val=None  # type: int
                         ):  # type: (...) -> GeotiffImage
        geotiff_image = GeotiffImage.init_from_numpy_array(image_data, geo_t, projection, nodata_val)
        return geotiff_image
