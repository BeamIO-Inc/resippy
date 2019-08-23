from __future__ import division

import gdal
import numpy as np
from numpy import ndarray

from resippy.image_objects.earth_overhead.abstract_earth_overhead_image import AbstractEarthOverheadImage
from resippy.image_objects.earth_overhead.agisoft_quick_preview.agisoft_quick_preview_metadata import AgisoftQuickPreviewMetadata
from resippy.image_objects.earth_overhead.agisoft_quick_preview.agisoft_quick_preview_point_calc import AgisoftQuickPreviewPointCalc
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.pinhole_camera import PinholeCamera


class AgisoftQuickPreviewImage(AbstractEarthOverheadImage):
    """Concrete implementations should initialize an image for reading/writing
    and should also set the image's metadata object and point calculator object"""

    def __init__(self):
        super(AgisoftQuickPreviewImage, self).__init__()
        self._dset = None

    @classmethod
    def init_from_file_and_nav(cls,
                               fname,  # type: str
                               sensor_lon_deg,
                               sensor_lat_deg,
                               sensor_alt_meters_msl,  # type: float
                               roll_degrees,
                               pitch_degrees,
                               yaw_degrees,
                               focal_length_mm,
                               pixel_pitch_microns,
                               ):
        agisoft_quick_preview_image = cls()
        agisoft_quick_preview_image.set_dset(gdal.Open(fname, gdal.GA_ReadOnly))
        metadata = AgisoftQuickPreviewMetadata._from_file(fname)
        npix_x = metadata.get_npix_x()
        npix_y = metadata.get_npix_y()
        point_calc = AgisoftQuickPreviewPointCalc.init_from_params(sensor_lon_deg,
                                                                   sensor_lat_deg,
                                                                   sensor_alt_meters_msl,
                                                                   roll_degrees,
                                                                   pitch_degrees,
                                                                   yaw_degrees,
                                                                   focal_length_mm,
                                                                   pixel_pitch_microns,
                                                                   npix_x,
                                                                   npix_y)
        agisoft_quick_preview_image.set_metadata(metadata)
        agisoft_quick_preview_image.set_point_calculator(point_calc)
        return agisoft_quick_preview_image

    def get_metadata(self):  # type: (...) -> AgisoftQuickPreviewMetadata
        return super(AgisoftQuickPreviewImage, self).get_metadata()

    def get_point_calculator(self):  # type: (...) -> AgisoftQuickPreviewPointCalc
        return super(AgisoftQuickPreviewImage, self).get_point_calculator()

    def set_dset(self,
                 dset  # type: gdal.Dataset
                 ):   # type: (...) -> None
        self._dset = dset

    def get_dset(self):  # type: (...) -> gdal.Dataset
        return self._dset

    def read_all_image_data_from_disk(self):  # type: (...) -> ndarray
        numpy_arr = []
        for bandnum in range(self.get_metadata().get_n_bands()):
            band = self._dset.GetRasterBand(bandnum + 1)
            numpy_arr.append(band.ReadAsArray())
        return np.stack(numpy_arr, axis=2)

    def read_band_from_disk(self,
                            band_number  # type: int
                            ):  # type: (...) -> ndarray
        band = self._dset.GetRasterBand(band_number + 1)
        return band.ReadAsArray()