from __future__ import division

from abc import ABC

import numpy as np
from numpy import ndarray
from resippy.image_objects.earth_overhead.abstract_earth_overhead_image import AbstractEarthOverheadImage
from shapely import geometry
import resippy.utils.photogrammetry_utils as photogram_utils
import scipy.interpolate as interp
from resippy.image_objects.image_factory import GeotiffImageFactory
from resippy.image_objects.earth_overhead.geotiff.geotiff_image import GeotiffImage
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.line_scanner import LineScannerPointCalc


class LineScannerImage(AbstractEarthOverheadImage):
    """Concrete implementations should initialize an image for reading/writing
    and should also set the image's metadata object and point calculator object"""

    def read_band_from_disk(self, band_number):
        pass

    def read_all_image_data_from_disk(self):
        pass

    def __init__(self):
        super(AbstractEarthOverheadImage, self).__init__()
        self._dset = None
        self.read_with_gdal = True
        self._point_calc = LineScannerPointCalc

    def ortho_image_patch(self,
                          input_image,
                          start_y,
                          start_x,
                          ny,
                          nx,
                          output_ny,
                          output_nx,
                          all_world_alts=None
                          ):  # type: (...) -> GeotiffImage

        point_calc = self.get_point_calculator()
        world_x_coords, world_y_coords = point_calc.get_all_world_xy_coords(alts=all_world_alts)
        patch_x_coords = world_x_coords[start_y:(start_y + ny), start_x:(start_x + nx)]
        patch_y_coords = world_y_coords[start_y:(start_y + ny), start_x:(start_x + nx)]
        patch_image_data = input_image[start_y:(start_y + ny), start_x:(start_x + nx)]

        minx = np.min(patch_x_coords)
        maxx = np.max(patch_x_coords)
        miny = np.min(patch_y_coords)
        maxy = np.max(patch_y_coords)
        p1 = geometry.Point(minx, maxy)
        p2 = geometry.Point(maxx, maxy)
        p3 = geometry.Point(maxx, miny)
        p4 = geometry.Point(minx, miny)
        pointList = [p1, p2, p3, p4, p1]
        poly = geometry.Polygon([[p.x, p.y] for p in pointList])
        geot = photogram_utils.world_poly_to_geo_t(poly, nx, ny)

        ground_x_grid, ground_y_grid = photogram_utils.create_ground_grid(minx, maxx, miny, maxy, output_nx,
                                                                          output_ny)

        x = np.ravel(world_x_coords)
        y = np.ravel(world_y_coords)
        z = np.ravel(patch_image_data)

        zi = interp.griddata((x, y), z, (ground_x_grid, ground_y_grid), method='nearest')
        gtiff_image = GeotiffImageFactory.from_numpy_array(zi, geot, point_calc.get_projection())
        return gtiff_image

    def ortho_entire_image(self,
                           input_image,
                           output_ny,
                           output_nx,
                           all_world_alts=None
                           ):  # type: (...) -> GeotiffImage
        point_calc = self.get_point_calculator()
        ny = len(point_calc.sensor_line_utm_northings)
        nx = len(point_calc.pixel_cross_track_angles_radians)
        return self.ortho_image_patch(input_image, 0, 0, ny, nx, output_ny, output_nx, all_world_alts=all_world_alts)
