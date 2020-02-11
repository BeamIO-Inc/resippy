from __future__ import division

from resippy.image_objects.earth_overhead.line_scanner.line_scanner_image import LineScannerImage
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.abstract_earth_overhead_point_calc import AbstractEarthOverheadPointCalc
from numpy import ndarray


class LineScannerImageFactory:
    @staticmethod
    def from_numpy_array_and_point_calc(image_data,  # type: ndarray
                                        point_calc,  # type: AbstractEarthOverheadPointCalc
                                        ):           # type: (...) -> LineScannerImage
        image = LineScannerImage()
        image.set_image_data(image_data)
        image.set_point_calculator(point_calc)
        return image

    @staticmethod
    def from_file(fname                 # type: str
                  ):                    # type: (...) -> LineScannerImage
        image = LineScannerImage.init_from_file(fname)
        return image
