from __future__ import division

from resippy.image_objects.earth_overhead.physical_camera.physical_camera_image import PhysicalCameraImage
from resippy.image_objects.earth_overhead.physical_camera.physical_camera_metadata import PhysicalCameraMetadata
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.abstract_earth_overhead_point_calc import AbstractEarthOverheadPointCalc
import gdal
from numpy import ndarray


class PhysicalCameraImageFactory:
    @staticmethod
    def from_gdal_file_and_single_point_calc(fname,                       # type: str
                                             point_calc,                  # type: AbstractEarthOverheadPointCalc
                                             ):       # type: (...) -> PhysicalCameraImage
        physical_camera_image = PhysicalCameraImage()
        physical_camera_image.set_gdal_dset(gdal.Open(fname, gdal.GA_ReadOnly))
        metadata = PhysicalCameraMetadata._from_gdal_file(fname)

        physical_camera_image.set_metadata(metadata)
        physical_camera_image.set_point_calculator(point_calc)
        return physical_camera_image

    @staticmethod
    def from_numpy_array_metadata_and_single_point_calc(image_data,  # type: ndarray
                                                        metadata,    # type: PhysicalCameraMetadata
                                                        point_calc,  # type: AbstractEarthOverheadPointCalc
                                                        ):  # type: (...) -> PhysicalCameraImage
        physical_camera = PhysicalCameraImage()
        physical_camera.set_image_data(image_data)
        physical_camera.set_metadata(metadata)
        physical_camera.set_point_calculator(point_calc)
        return physical_camera
