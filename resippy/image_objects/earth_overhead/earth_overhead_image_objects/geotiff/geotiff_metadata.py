from __future__ import division

from resippy.image_objects.abstract_image_metadata import AbstractImageMetadata
import gdal


class GeotiffMetadata(AbstractImageMetadata):
    @classmethod
    def _from_file(cls,
                   fname  # type: str
                   ):  # type: (...) -> GeotiffMetadata
        geotiff_metadata = cls()
        dset = gdal.Open(fname)
        geotiff_metadata.set_npix_x(dset.RasterXSize)
        geotiff_metadata.set_npix_y(dset.RasterYSize)
        geotiff_metadata.set_n_bands(dset.RasterCount)
        geotiff_metadata.set_gdal_datatype(dset.GetRasterBand(1).DataType)
        gdal_nodata_val = dset.GetRasterBand(1).GetNoDataValue()
        geotiff_metadata.set_nodata_val(gdal_nodata_val)
        dset = None
        return geotiff_metadata

    def set_gdal_datatype(self,
                          dtype  # type: int
                          ):  # type: (...) -> None
        self.metadata_dict["gdal_datatype"] = dtype

    def get_gdal_datatype(self):  # type: (...) -> int
        return self.metadata_dict["gdal_datatype"]
