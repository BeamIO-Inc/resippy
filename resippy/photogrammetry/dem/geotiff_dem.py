from __future__ import division

from resippy.photogrammetry.dem.abstract_dem import AbstractDem

from resippy.image_objects.earth_overhead.earth_overhead_image_objects.geotiff.geotiff_image import GeotiffImage, \
    reproject_vertical_datum
import numpy as np
from numpy import ndarray


class GeotiffDem(AbstractDem):
    def __init__(self):
        self.dem_data = None
        self.gtiff = GeotiffImage
        self.nodata_value = None
        self.interpolation_method = 'bilinear'

    def set_interpolation_to_nearest(self):
        self.interpolation_method = 'nearest'

    def set_interpolation_to_bilinear(self):
        self.interpolation_method = 'bilinear'

    def get_interpolation_method(self):     # type: (...) -> str
        return self.interpolation_method

    def convert_reference(self,
                          dst_fname,  # type: str
                          dst_epsg_code  # type: int
                          ):  # type: (...) -> None
        new_gtiff = reproject_vertical_datum(self.gtiff, dst_fname, dst_epsg_code)
        self.gtiff.close_image()
        self.set_geotiff_image(new_gtiff)

    def remove_nodata_values(self,
                             nodata_value  # type: float
                             ):  # type: (...) -> None
        self.nodata_value = nodata_value
        if self.nodata_value is not None:
            if self.dem_data is not None:
                self.dem_data[np.where(self.dem_data == nodata_value)] = np.nan

    def set_geotiff_image(self,
                          geotiff_image  # type: GeotiffImage
                          ):  # type: (...) -> None
        self.gtiff = geotiff_image
        self.dem_data = np.squeeze(self.gtiff.read_all_image_data_from_disk())
        self.set_projection(self.gtiff.get_point_calculator().get_projection())
        self.remove_nodata_values(self.nodata_value)

    def get_elevations(self,
                       world_x,  # type: ndarray
                       world_y,  # type: ndarray
                       world_proj=None,     # type: Proj
                       ):  # type: (...) -> ndarray

        pixel_locs_x, pixel_locs_y = self.gtiff.get_point_calculator().lon_lat_alt_to_pixel_x_y(world_x, world_y,
                                                                                                alts=None,
                                                                                                world_proj=world_proj)
        world_xyz_is_2d = False
        if np.ndim(world_x) == 2:
            world_xyz_is_2d = True
            nx = np.shape(world_x)[1]
            ny = np.shape(world_x)[0]
            pixel_locs_x = np.reshape(pixel_locs_x, nx * ny)
            pixel_locs_y = np.reshape(pixel_locs_y, nx * ny)

        if type(pixel_locs_x) is float:
            pixel_locs_x = np.array(pixel_locs_x)
            pixel_locs_y = np.array(pixel_locs_y)

        # TODO implement other interpoloation methods, right now we just grab the closest pixels
        if self.get_interpolation_method() is 'nearest':
            pixel_locs_x = np.clip(pixel_locs_x.astype(int), a_min=0, a_max=self.gtiff.get_metadata().get_npix_x()-1)
            pixel_locs_y = np.clip(pixel_locs_y.astype(int), a_min=0, a_max=self.gtiff.get_metadata().get_npix_y()-1)
            elevations = self.dem_data[pixel_locs_y, pixel_locs_x]
        elif self.get_interpolation_method() is 'bilinear':
            # Do a bilinear interpolation between closest pixels by taking a weighted average
            pixel_locs_x_1 = np.clip(pixel_locs_x.astype(int), a_min=0, a_max=self.gtiff.get_metadata().get_npix_x() - 1)
            pixel_locs_y_1 = np.clip(pixel_locs_y.astype(int), a_min=0, a_max=self.gtiff.get_metadata().get_npix_y() - 1)

            pixel_locs_x_2 = np.clip(np.ceil(pixel_locs_x).astype(int), a_min=0, a_max=self.gtiff.get_metadata().get_npix_x() - 1)
            pixel_locs_y_2 = np.clip(np.ceil(pixel_locs_y).astype(int), a_min=0, a_max=self.gtiff.get_metadata().get_npix_y() - 1)

            elevations_y1x1 = self.dem_data[pixel_locs_y_1, pixel_locs_x_1]
            elevations_y1x2 = self.dem_data[pixel_locs_y_1, pixel_locs_x_2]
            elevations_y2x1 = self.dem_data[pixel_locs_y_2, pixel_locs_x_1]
            elevations_y2x2 = self.dem_data[pixel_locs_y_2, pixel_locs_x_2]

            x_distances_1 = pixel_locs_x - pixel_locs_x_1
            x_distances_2 = pixel_locs_x_2 - pixel_locs_x
            y_distances_1 = pixel_locs_y - pixel_locs_y_1
            y_distances_2 = pixel_locs_y_2 - pixel_locs_y

            distances_y1x1 = np.sqrt(np.square(y_distances_1) + np.square(x_distances_1))
            distances_y1x2 = np.sqrt(np.square(y_distances_1) + np.square(x_distances_2))
            distances_y2x1 = np.sqrt(np.square(y_distances_2) + np.square(x_distances_1))
            distances_y2x2 = np.sqrt(np.square(y_distances_2) + np.square(x_distances_2))

            elevations_arr = np.array((elevations_y1x1, elevations_y1x2, elevations_y2x1, elevations_y2x2))
            weights_arr = np.array((distances_y1x1, distances_y1x2, distances_y2x1, distances_y2x2))

            elevations = np.average(elevations_arr, weights=weights_arr, axis=0,)

        else:
            TypeError("interpolation must be set to either 'nearest' or 'bilinear' "
                      "when getting elevations from a GeoTiff DEM")

        # unflatten world_xyz arrays if the original inputs were 2d
        if world_xyz_is_2d:
            elevations = np.reshape(elevations, (ny, nx))
        else:
            elevations = np.squeeze(elevations)
        return elevations

    def get_highest_alt(self):  # type: (...) -> float
        return float(np.nanmax(self.dem_data))

    def get_lowest_alt(self):  # type: (...) -> float
        return float(np.nanmin(self.dem_data))
