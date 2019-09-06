from __future__ import division

from resippy.image_objects.abstract_image_metadata import AbstractImageMetadata
import gdal
import os


class PhysicalCameraMetadata(AbstractImageMetadata):
    @classmethod
    def _from_gdal_file(cls,
                        fname   # type: str
                        ):      # type: (...) -> PhysicalCameraMetadata
        physical_camera_metadata = cls()
        physical_camera_metadata.set_image_name(os.path.splitext(fname)[0])
        dset = gdal.Open(fname)
        physical_camera_metadata.set_npix_x(dset.RasterXSize)
        physical_camera_metadata.set_npix_y(dset.RasterYSize)
        physical_camera_metadata.set_n_bands(dset.RasterCount)
        physical_camera_metadata.set_gdal_datatype(dset.GetRasterBand(1).DataType)
        gdal_nodata_val = dset.GetRasterBand(1).GetNoDataValue()
        physical_camera_metadata.set_nodata_val(gdal_nodata_val)
        dset = None
        return physical_camera_metadata

    def set_gdal_datatype(self,
                          dtype  # type: int
                          ):  # type: (...) -> None
        self.metadata_dict["gdal_datatype"] = dtype

    def get_gdal_datatype(self):  # type: (...) -> int
        return self.metadata_dict["gdal_datatype"]
