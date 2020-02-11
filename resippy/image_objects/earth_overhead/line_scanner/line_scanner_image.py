from __future__ import division

import gdal
import numpy as np
import scipy.interpolate as interp
from shapely import geometry
from resippy.image_objects.earth_overhead.abstract_earth_overhead_image import AbstractEarthOverheadImage
import resippy.utils.photogrammetry_utils as photogram_utils
from resippy.image_objects.envi.envi_metadata import EnviMetadata
from resippy.image_objects.image_factory import GeotiffImageFactory
from resippy.image_objects.earth_overhead.geotiff.geotiff_image import GeotiffImage
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.line_scanner import LineScannerPointCalc


class LineScannerImage(AbstractEarthOverheadImage):
    """Concrete implementations should initialize an image for reading/writing
    and should also set the image's metadata object and point calculator object"""

    def __init__(self):
        super(AbstractEarthOverheadImage, self).__init__()
        self._dset = None
        self.read_with_gdal = True
        self._point_calc = LineScannerPointCalc

    @classmethod
    def init_from_file(cls,
                       image_file_path      # type: str
                       ):                   # type: (...) -> LineScannerImage
        image = cls()
        image.set_dset(gdal.Open(image_file_path, gdal.GA_ReadOnly))

        metadata = EnviMetadata.init_from_file(image_file_path)
        image.set_metadata(metadata)
        return image

    def read_band_from_disk(self,
                            band_number  # type: int
                            ):           # type: (...) -> np.ndarray
        band = self._dset.GetRasterBand(band_number + 1)
        return band.ReadAsArray()

    def read_all_image_data_from_disk(self
                                      ):    # type: (...) -> np.ndarray
        numpy_arr = []
        for bandnum in range(self.get_metadata().get_n_bands()):
            band = self._dset.GetRasterBand(bandnum + 1)
            numpy_arr.append(band.ReadAsArray())
        return np.stack(numpy_arr, axis=2)

    def set_dset(self,
                 dset   # type: gdal.Dataset
                 ):     # type: (...) -> None
        self._dset = dset

    def ortho_image_patch(self,
                          start_y,
                          start_x,
                          ny,
                          nx,
                          output_ny,
                          output_nx,
                          band_num=50,
                          all_world_alts=None
                          ):  # type: (...) -> GeotiffImage
        input_image = self.read_band_from_disk(band_num)
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
        point_list = [p1, p2, p3, p4, p1]
        poly = geometry.Polygon([[p.x, p.y] for p in point_list])
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
                           output_ny,
                           output_nx,
                           band_num=50,
                           all_world_alts=None
                           ):  # type: (...) -> GeotiffImage
        point_calc = self.get_point_calculator()
        ny = len(point_calc.sensor_line_utm_northings)
        nx = len(point_calc.pixel_cross_track_angles_radians)
        return self.ortho_image_patch(0, 0, ny, nx, output_ny, output_nx, band_num=band_num,
                                      all_world_alts=all_world_alts)
