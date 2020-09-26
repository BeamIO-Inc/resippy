from __future__ import division

from resippy.image_objects.earth_overhead.linescanner.linescanner_image import LinescannerImage
from resippy.image_objects.earth_overhead.linescanner.linescanner_metadata import LinescannerMetadata
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.line_scanner_point_calc import LineScannerPointCalc
from numpy import ndarray


class LinescannerImageFactory:
    @staticmethod
    def from_numpy_array_metadata_and_point_calc(image_data,  # type: ndarray
                                                 metadata,    # type: LinescannerMetadata
                                                 point_calc,  # type: LineScannerPointCalc
                                                 ):  # type: (...) -> PhysicalCameraImage
        linescanner = LinescannerImage()
        linescanner.set_image_data(image_data)
        linescanner.set_metadata(metadata)
        linescanner.set_point_calculator(point_calc)
        return linescanner
