from __future__ import division

from resippy.image_objects.earth_overhead.abstract_earth_overhead_point_calc import AbstractEarthOverheadPointCalc
import gdal
import osr
from pyproj import Proj
from shapely.geometry.geo import box, Polygon
import numpy as np
from numpy import ndarray


class GeotiffPointCalc(AbstractEarthOverheadPointCalc):

    def __init__(self):
        self._geo_t = None
        self._inv_geo_t = None
        self._npix_x = None
        self._npix_y = None

    @classmethod
    def init_from_file(cls,
                       fname  # type: str
                       ):  # type: (...) -> GeotiffPointCalc
        point_calc = cls()
        dset = gdal.Open(fname)
        point_calc.set_geot(np.array(dset.GetGeoTransform()))
        point_calc.set_npix_x(dset.RasterXSize)
        point_calc.set_npix_y(dset.RasterYSize)
        srs_wkt = dset.GetProjection()
        srs_converter = osr.SpatialReference()  # makes an empty spatial ref object
        srs_converter.ImportFromWkt(srs_wkt)  # populates the spatial ref object with our WKT SRS
        point_calc.set_projection(Proj(srs_converter.ExportToProj4()))
        dset = None
        return point_calc

    def set_geot(self,
                 geo_t  # type: list
                 ):  # type: (...) -> list
        self._geo_t = geo_t
        self._inv_geo_t = gdal.InvGeoTransform(geo_t)

    def get_geot(self):  # type: (...) -> ndarray
        return self._geo_t

    def get_inv_geot(self):  # type: (...) -> ndarray
        return self._inv_geo_t

    def set_npix_x(self,
                   image_npix_x  # type: int
                   ):  # type: (...) -> None
        self._npix_x = image_npix_x

    def set_npix_y(self,
                   image_npix_y  # type: int
                   ):  # type: (...) -> None
        self._npix_y = image_npix_y

    def _lon_lat_alt_to_pixel_x_y_native(self,
                                         lons,  # type: ndarray
                                         lats,  # type: ndarray
                                         alts=None,  # type: ndarray
                                         band=None  # type: int
                                         ):  # type: (...) -> (ndarray, ndarray)
        x = self._inv_geo_t[0] + self._inv_geo_t[1] * lons + self._inv_geo_t[2] * lats
        y = self._inv_geo_t[3] + self._inv_geo_t[4] * lons + self._inv_geo_t[5] * lats
        return x, y

    def _pixel_x_y_alt_to_lon_lat_native(self,
                                         x_pixels,  # type: ndarray
                                         y_pixels,  # type: ndarray
                                         alts=None,  # type: ndarray
                                         band=None  # type: ndarray
                                         ):  # type: (...) -> (ndarray, ndarray)
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
