from __future__ import division

from resippy.image_objects.earth_overhead.line_scanner.line_scanner_image import LineScannerImage
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.abstract_earth_overhead_point_calc import AbstractEarthOverheadPointCalc
from numpy import ndarray


class LineScannerImageFactory:
    @staticmethod
    def from_numpy_array_and_point_calc(image_data,  # type: ndarray
                                        point_calc,  # type: AbstractEarthOverheadPointCalc
                                        ):  # type: (...) -> Line
        physical_camera = LineScannerImage()
        physical_camera.set_image_data(image_data)
        physical_camera.set_point_calculator(point_calc)
        return physical_camera
