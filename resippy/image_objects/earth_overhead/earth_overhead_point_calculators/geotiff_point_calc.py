from __future__ import division

from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.abstract_earth_overhead_point_calc \
    import AbstractEarthOverheadPointCalc
import gdal
import osr
from pyproj import Proj
from shapely.geometry.geo import box, Polygon
import numpy as np
from numpy import ndarray


class GeotiffPointCalc(AbstractEarthOverheadPointCalc):
    """
    This is a concrete implementation of AbstractEarthOverheadPointCalc for a geotiff point calculator.
    """

    def __init__(self):
        self._geo_t = None
        self._inv_geo_t = None
        self._npix_x = None
        self._npix_y = None
        self._bands_coregistered = True

    @classmethod
    def init_from_file(cls,
                       fname  # type: str
                       ):  # type: (...) -> GeotiffPointCalc
        """
        This is a class method that returns an initialized GeotiffPointCalc object from an input file.
        :param fname: filename of geotiff image file
        :return: initialized Geotiff point calculator
        """
        point_calc = cls()
        dset = gdal.Open(fname)
        point_calc.set_geot(np.array(dset.GetGeoTransform()))
        point_calc.set_npix_x(dset.RasterXSize)
        point_calc.set_npix_y(dset.RasterYSize)
        srs_wkt = dset.GetProjection()
        srs_converter = osr.SpatialReference()  # makes an empty spatial ref object
        srs_converter.ImportFromWkt(srs_wkt)  # populates the spatial ref object with our WKT SRS
        point_calc.set_projection(Proj(srs_converter.ExportToProj4(), preserve_units=True))
        dset = None
        return point_calc

    def set_geot(self,
                 geo_t  # type: list
                 ):  # type: (...) -> None
        """
        sets the geotiffs geotransform and inverse geotransform
        :param geo_t: geotransform specified by gdal's documentation.
        :return: None
        """
        self._geo_t = geo_t
        self._inv_geo_t = gdal.InvGeoTransform(geo_t)

    def get_geot(self):  # type: (...) -> ndarray
        """
        Returns the geotransform for the point calculator
        :return: list of geotransform parameters
        """
        return self._geo_t

    def get_inv_geot(self):  # type: (...) -> ndarray
        """
        Returns the inverse geotransform for the point calculator
        :return: list of inverse geotransform parameters
        """
        return self._inv_geo_t

    def set_npix_x(self,
                   image_npix_x  # type: int
                   ):  # type: (...) -> None
        """
        Sets the number of x pixels for the image.  This is generally useful for computing image geographic bounds
        :param image_npix_x: number of x pixels, integer
        :return: None
        """
        self._npix_x = image_npix_x

    def set_npix_y(self,
                   image_npix_y  # type: int
                   ):  # type: (...) -> None
        """
        Sets the number of y pixels for the image.  This is generally useful for computing image geographic bounds
        :param image_npix_y: number of x pixels, integer
        :return: None
        """
        self._npix_y = image_npix_y

    def _lon_lat_alt_to_pixel_x_y_native(self,
                                         lons,  # type: ndarray
                                         lats,  # type: ndarray
                                         alts=None,  # type: ndarray
                                         band=None  # type: int
                                         ):  # type: (...) -> (ndarray, ndarray)
        """
        Uses the point calculator's inverse geotransform parameters to calculate pixel locations from
        longitude, latitude. Altitude is ignored in this case.
        See documentation for AbstractEarthOverheadPointCalc.
        :param lons:
        :param lats:
        :param alts:
        :param band:
        :return:
        """
        x = self._inv_geo_t[0] + self._inv_geo_t[1] * lons + self._inv_geo_t[2] * lats
        y = self._inv_geo_t[3] + self._inv_geo_t[4] * lons + self._inv_geo_t[5] * lats
        return x, y

    def _pixel_x_y_alt_to_lon_lat_native(self,
                                         x_pixels,  # type: ndarray
                                         y_pixels,  # type: ndarray
                                         alts=None,  # type: ndarray
                                         band=None  # type: ndarray
                                         ):  # type: (...) -> (ndarray, ndarray)
        """
        Uses the point calculator's geotransform parameters to calculate longitude, latitude locations from the image
        pixel x y locations.  Altitude is ignored in this case.
        See documentation for AbstractEarthOverheadPointCalc.
        :param x_pixels:
        :param y_pixels:
        :param alts:
        :param band:
        :return:
        """
        lons = self._geo_t[0] + x_pixels * self._geo_t[1] + y_pixels * self._geo_t[2]
        lats = self._geo_t[3] + x_pixels * self._geo_t[4] + y_pixels * self._geo_t[5]
        return lons, lats

    def get_gdal_projection_wkt(self):  # type: (...) -> str
        srs = osr.SpatialReference()
        srs.ImportFromProj4(self.get_projection().srs)
        return srs.ExportToWkt()

    def get_world_extent_native(self):  # type: (...) -> Polygon
        lon_ul, lat_ul = self._pixel_x_y_alt_to_lon_lat_native(0, 0)
        lon_br, lat_br = self._pixel_x_y_alt_to_lon_lat_native(self._npix_x, self._npix_y)
        world_poly = box(lon_ul, lat_br, lon_br, lat_ul)
        return world_poly
