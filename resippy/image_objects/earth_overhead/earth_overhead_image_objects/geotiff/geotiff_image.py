from __future__ import division

import gdal
import numpy as np
import osr
from numpy import ndarray
from osgeo import gdal_array
from pyproj import Proj

from resippy.image_objects.earth_overhead.abstract_earth_overhead_image import AbstractEarthOverheadImage
from resippy.image_objects.earth_overhead.earth_overhead_image_objects.geotiff.geotiff_metadata import GeotiffMetadata
from resippy.image_objects.earth_overhead.earth_overhead_image_objects.geotiff.geotiff_point_calc import GeotiffPointCalc


class GeotiffImage(AbstractEarthOverheadImage):
    """Concrete implementations should initialize an image for reading/writing
    and should also set the image's metadata object and point calculator object"""

    def __init__(self):
        super(GeotiffImage, self).__init__()
        self._dset = None

    @classmethod
    def init_from_file(cls,
                       fname  # type: str
                       ):  # type: (...) -> GeotiffImage
        geotiff_image = cls()
        geotiff_image.set_dset(gdal.Open(fname, gdal.GA_ReadOnly))
        metadata = GeotiffMetadata._from_file(fname)
        point_calc = GeotiffPointCalc.init_from_file(fname)
        geotiff_image.set_metadata(metadata)
        geotiff_image.set_point_calculator(point_calc)
        return geotiff_image

    @classmethod
    def init_from_numpy_array(cls,
                              image_data,  # type: ndarray
                              geo_t,  # type: list
                              projection,  # type: Proj
                              nodata_val=None  # type: int
                              ):  # type: (...) -> GeotiffImage
        geotiff_image = cls()
        geotiff_image.set_image_data(image_data)

        image_shape = image_data.shape
        nx = image_shape[1]
        ny = image_shape[0]
        nbands = image_shape[2]
        gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(image_data.dtype)
        metadata = GeotiffMetadata()
        metadata.set_npix_x(nx)
        metadata.set_npix_y(ny)
        metadata.set_n_bands(nbands)
        metadata.set_gdal_datatype(gdal_dtype)
        metadata.set_nodata_val(nodata_val)
        point_calc = GeotiffPointCalc()
        point_calc.set_npix_x(nx)
        point_calc.set_npix_y(ny)
        point_calc.set_geot(geo_t)
        point_calc.set_projection(projection)
        geotiff_image.set_metadata(metadata)
        geotiff_image.set_point_calculator(point_calc)
        return geotiff_image

    def get_metadata(self):  # type: (...) -> GeotiffMetadata
        return super(GeotiffImage, self).get_metadata()

    def get_point_calculator(self):  # type: (...) -> GeotiffPointCalc
        return super(GeotiffImage, self).get_point_calculator()

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

    def get_gdal_mem_datset(self):  # type: (...) -> gdal.Dataset
        driver = gdal.GetDriverByName('MEM')
        dtype = self.get_metadata().get_gdal_datatype()
        dataset = driver.Create(
            '',
            self.get_metadata().get_npix_x(),
            self.get_metadata().get_npix_y(),
            self.get_metadata().get_n_bands(),
            dtype)

        dataset.SetGeoTransform(self.get_point_calculator().get_geot())
        dataset.SetProjection(self.get_point_calculator().get_gdal_projection_wkt())
        if self.get_image_data() is None:
            self.set_image_data(self.read_all_image_data_from_disk())
        for band in range(self.get_metadata().get_n_bands()):
            dataset.GetRasterBand(band + 1).WriteArray(self.get_image_band(band))
            if self.get_metadata().get_nodata_val() is not None:
                dataset.GetRasterBand(band + 1).SetNoDataValue(int(self.get_metadata().get_nodata_val()))
        return dataset

    def write_to_disk(self,
                      fname,  # type: str
                      compression="LZW"  # type: str
                      ):  # type: (...) -> None
        driver = gdal.GetDriverByName('GTiff')
        dtype = self.get_metadata().get_gdal_datatype()
        ops = ['COMPRESS=' + compression, "INTERLEAVE=BAND", "TILED=YES"]
        dataset = driver.Create(
            fname,
            self.get_metadata().get_npix_x(),
            self.get_metadata().get_npix_y(),
            self.get_metadata().get_n_bands(),
            dtype,
            ops)
        dataset.SetGeoTransform(self.get_point_calculator().get_geot())
        dataset.SetProjection(self.get_point_calculator().get_gdal_projection_wkt())
        if self.get_image_data() is None:
            self.set_image_data(self.read_all_image_data_from_disk())
        for band in range(self.get_metadata().get_n_bands()):
            dataset.GetRasterBand(band + 1).WriteArray(self.get_image_band(band))
            if self.get_metadata().get_nodata_val() is not None:
                dataset.GetRasterBand(band + 1).SetNoDataValue(int(self.get_metadata().get_nodata_val()))
        dataset.FlushCache()
        dataset = None

    def close_image(self):  # type: (...) -> None
        self._dset = None

    def __del__(self):
        self._dset = None


def gdalwarp_geoTiffImage(image,        # type: GeotiffImage
                          dst_fname,    # type: str
                          warp_ops      # type: gdal.WarpOptions
                          ):            # type: (...) -> GeotiffImage

    src_dataset = image.get_dset()
    if src_dataset is None:
        src_dataset = image.get_gdal_mem_datset()

    dst_dataset = gdal.Warp(dst_fname, src_dataset, options=warp_ops)
    src_dataset = None
    dst_dataset = None

    return GeotiffImage.init_from_file(dst_fname)


# for a comprehensive list see: https://vdatum.noaa.gov/docs/datums.html#verticaldatum
def reproject_vertical_datum(geotiff_image,     # type: GeotiffImage
                             dst_fname,         # type: str
                             datum_epsg_code    # type: str
                             ):                 # type: (...) -> GeotiffImage
    proj4_srs = geotiff_image.get_point_calculator().get_projection().srs
    srs = osr.SpatialReference()
    srs.ImportFromProj4(proj4_srs)
    srs.AutoIdentifyEPSG()
    dst_epsg = 'EPSG:' + str(srs.GetAuthorityCode(None)) + "+" + str(datum_epsg_code)

    ops = gdal.WarpOptions(dstSRS=dst_epsg)
    return gdalwarp_geoTiffImage(geotiff_image, dst_fname, ops)
