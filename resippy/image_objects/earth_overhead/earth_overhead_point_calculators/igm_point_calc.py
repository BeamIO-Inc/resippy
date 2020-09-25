from __future__ import division

from typing import Union
import numbers
import numpy
from numpy import ndarray
from pyproj import Proj
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.abstract_earth_overhead_point_calc \
    import AbstractEarthOverheadPointCalc
from scipy.spatial.ckdtree import cKDTree


class IGMPointCalc(AbstractEarthOverheadPointCalc):
    def __init__(self,
                 lon_array,  # type: ndarray
                 lat_array,  # type: ndarray
                 alt_array,  # type: Union[ndarray, float]
                 projection,  # type:  Proj
                 ):
        self._lons = lon_array
        self._lats = lat_array
        self._npix_y, self._npix_x = lon_array.shape
        self.set_projection(projection)
        if isinstance(alt_array, numbers.Number):
            self._alts = numpy.zeros_like(lon_array)
        else:
            self._alts = alt_array
        self._bands_coregistered = True

        self._lons_1d = None  # type: ndarray
        self._lats_1d = None  # type: ndarray
        self._kd_tree = None  # type: KDTree
        self.set_approximate_lon_lat_center(lon_array[int(self._npix_y/2), int(self._npix_x/2)],
                                            lat_array[int(self._npix_y/2), int(self._npix_x/2)])

    @property
    def lon_image(self):
        return self._lons

    @property
    def lat_image(self):
        return self._lats

    @property
    def alt_image(self):
        return self._alts

    def _pixel_x_y_alt_to_lon_lat_native(self, pixel_xs, pixel_ys, alts=None, band=None):
        return self._lons[pixel_ys, pixel_xs], self._lats[pixel_ys, pixel_xs]

    def _lon_lat_alt_to_pixel_x_y_native(self, lons, lats, alts, band=None):
        if self._lons_1d is None:
            self._lons_1d = numpy.ravel(self._lons)
        if self._lats_1d is None:
            self._lats_1d = numpy.ravel(self._lats)
        if self._kd_tree is None:
            kd_tree_data = numpy.transpose((self._lons_1d, self._lats_1d))
            self._kd_tree = cKDTree(kd_tree_data)

        distances, indices_1d = self._kd_tree.query(numpy.asarray((lons, lats)).transpose(), 6)
        indices_2d = numpy.unravel_index(indices_1d, (self._npix_y, self._npix_x))

        # Perform the interpolation here using an inverse distance weighted method
        distances[numpy.isclose(distances, 0)] = 0.00000001
        inv_distances = 1 / distances
        inv_distances_sum = numpy.sum(1 / distances, axis=1)
        interpolated_y_vals = numpy.sum(inv_distances * indices_2d[0], axis=1) / inv_distances_sum
        interpolated_x_vals = numpy.sum(inv_distances * indices_2d[1], axis=1) / inv_distances_sum
        return interpolated_x_vals, interpolated_y_vals

