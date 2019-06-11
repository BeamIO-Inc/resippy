from __future__ import division

import abc
from numpy import ndarray
from pyproj import Proj
from pyproj import transform as proj_transform
import numpy as np
import numbers
from resippy.photogrammetry.dem.abstract_dem import AbstractDem
import resippy.utils.image_utils.image_utils as image_utils
from six import add_metaclass


@add_metaclass(abc.ABCMeta)
class AbstractEarthOverheadPointCalc:
    """Concrete implementations should initialize an image for reading/writing
    and should also set the image's metadata object and point calculator object"""

    def __init__(self):
        self._lon_lat_center_approximate = None
        self._projection = None
        self._bands_coregistered = True

    @abc.abstractmethod
    def _lon_lat_alt_to_pixel_x_y_native(self,
                                         lons,  # type: ndarray
                                         lats,  # type: ndarray
                                         alts,  # type: ndarray
                                         band=None  # type: int
                                         ):  # type: (...) -> (ndarray, ndarray)
        pass

    @abc.abstractmethod
    def _pixel_x_y_alt_to_lon_lat_native(self,
                                         pixel_xs,  # type: ndarray
                                         pixel_ys,  # type: ndarray
                                         alts=None,  # type: ndarray
                                         band=None  # type: int
                                         ):  # type: (...) -> (ndarray, ndarray)
        pass

    def lon_lat_alt_to_pixel_x_y(self,
                                 lons,  # type: ndarray
                                 lats,  # type: ndarray
                                 alts,  # type: ndarray
                                 world_proj=None,  # type: Proj
                                 band=None  # type: int
                                 ):  # type: (...) -> (ndarray, ndarray)

        # check for some errors up front
        if alts is None:
            alts = 0
        # standardize inputs, make everything 1 dimensional ndarrays
        lons_is_number = isinstance(lons, numbers.Number)
        lats_is_number = isinstance(lats, numbers.Number)
        alts_is_number = isinstance(alts, numbers.Number)
        if lons_is_number or lats_is_number:
            lons = np.array([lons])
            lats = np.array([lats])
        if alts_is_number:
            alts = np.zeros(lons.shape) + alts
        world_xyz_is_2d = False
        # auto-detect if world x-y-z arrays are 2d and flatten world_x and world_y arrays they are are 2d.
        # This is done to make all the vector math less complicated and keep it fast without needing to use loops
        if np.ndim(lons) == 2:
            world_xyz_is_2d = True
            nx = np.shape(lons)[1]
            ny = np.shape(lons)[0]
            lons = np.reshape(lons, nx * ny)
            lats = np.reshape(lats, nx * ny)
            alts = np.reshape(alts, nx * ny)

        # now actually do the calculations with everything in a standard form
        if world_proj is None:
            world_proj = self.get_projection()
        if world_proj.srs != self.get_projection().srs:
            lons, lats, alts = proj_transform(world_proj, self.get_projection(), lons, lats, alts)
        pixel_coords = self._lon_lat_alt_to_pixel_x_y_native(lons, lats, alts, band)

        if lons_is_number or lats_is_number:
            pixel_coords = pixel_coords[0][0], pixel_coords[1][0]

        # now transform everything back if it wasn't in a standard form coming in
        # unflatten world_xyz arrays if the original inputs were 2d
        if world_xyz_is_2d:
            pixel_coords_x_2d = np.reshape(pixel_coords[0], (ny, nx))
            pixel_coords_y_2d = np.reshape(pixel_coords[1], (ny, nx))
            return pixel_coords_x_2d, pixel_coords_y_2d

        return pixel_coords

    def pixel_x_y_alt_to_lon_lat(self,
                                 pixel_xs,  # type: ndarray
                                 pixel_ys,  # type: ndarray
                                 alts,  # type: ndarray
                                 world_proj=None,  # type: Proj
                                 band=None,  # type: int
                                 pixel_error_threshold=0.01,  # type: float
                                 max_iter=1000,  # type: int
                                 ):  # type: (...) -> (ndarray, ndarray)
        if world_proj is None:
            world_proj = self.get_projection()
        if self._pixel_x_y_alt_to_lon_lat_native(pixel_xs, pixel_ys, alts, band) is not None:
            native_lons, native_lats = self._pixel_x_y_alt_to_lon_lat_native(pixel_xs, pixel_ys, alts, band=band)
        else:
            native_lons, native_lats = \
                self._pixel_x_y_alt_to_lon_lat_native_solver(pixel_xs,
                                                             pixel_ys,
                                                             alts,
                                                             band=band,
                                                             max_pixel_error=pixel_error_threshold,
                                                             max_iter=max_iter)
        if world_proj.srs != self.get_projection().srs:
            lons, lats = proj_transform(self.get_projection(), world_proj, native_lons, native_lats)
            return lons, lats
        else:
            return native_lons, native_lats

    def _pixel_x_y_alt_to_lon_lat_native_solver(self,
                                                pixel_xs,  # type: ndarray
                                                pixel_ys,  # type: ndarray
                                                alts,  # type: ndarray
                                                d_lon=None,  # type: float
                                                d_lat=None,  # type: float
                                                band=None,  # type: int
                                                max_pixel_error=0.01,  # type: float
                                                max_iter=1000,  # type: int
                                                ):  # type: (...) -> (ndarray, ndarray)

        n_pixels = np.shape(pixel_xs)
        approximate_lon, approximate_lat = self.get_approximate_lon_lat_center()
        # initial lons and lats to all be the approximate center of the image
        lons, lats = np.zeros(n_pixels) + approximate_lon, np.zeros(n_pixels) + approximate_lat

        if d_lon is None or d_lat is None:
            d_pixel = 1
            # we want the delta to be on the order of 1 pixel or so, maybe change this later to scale with the errors
            machine_eps = np.finfo(lons.dtype).eps
            machine_max_val = np.finfo(lons.dtype).max
            machine_val_cutoff = machine_max_val / 4.0
            float_nums = []
            current_num = machine_eps
            while current_num < machine_val_cutoff:
                float_nums.append(current_num)
                current_num = current_num * 2
            float_nums = np.array(float_nums)
            machine_lons = np.zeros(np.shape(float_nums)) + approximate_lon
            machine_lats = np.zeros(np.shape(float_nums)) + approximate_lat
            machine_d_lons = machine_lons + float_nums
            machine_d_lats = machine_lats + float_nums
            machine_alt = np.average(alts)

            machine_pixel_lon_lat = \
                self.lon_lat_alt_to_pixel_x_y(approximate_lon, approximate_lat, machine_alt, band=band)

            machine_lon_pixels_x, machine_lon_pixels_y = \
                self.lon_lat_alt_to_pixel_x_y(machine_d_lons, machine_lats, machine_alt, band=band)
            machine_lat_pixels_x, machine_lat_pixels_y = \
                self.lon_lat_alt_to_pixel_x_y(machine_lons, machine_d_lats, machine_alt, band=band)

            machine_pixels_lon_x_diff = machine_lon_pixels_x - machine_pixel_lon_lat[0]
            machine_pixels_lon_y_diff = machine_lon_pixels_y - machine_pixel_lon_lat[1]

            machine_pixels_lat_x_diff = machine_lat_pixels_x - machine_pixel_lon_lat[0]
            machine_pixels_lat_y_diff = machine_lat_pixels_y - machine_pixel_lon_lat[1]

            # find the first index where a shift in longitude is greater than 1 pixel
            lon_gt1_pixel_index = (np.where(np.square(machine_pixels_lon_x_diff) +
                                            np.square(machine_pixels_lon_y_diff) > np.square(d_pixel))[0])[0]
            d_lon = machine_d_lons[lon_gt1_pixel_index] - approximate_lon

            # find the first index where a shift in longitude is greater than 1 pixel
            lat_gt1_pixel_index = (np.where(np.square(machine_pixels_lat_x_diff) +
                                            np.square(machine_pixels_lat_y_diff) > np.square(d_pixel))[0])[0]
            d_lat = machine_d_lats[lat_gt1_pixel_index] - approximate_lat

        for i in range(max_iter):
            pixel_x_estimate, pixel_y_estimate = self.lon_lat_alt_to_pixel_x_y(lons, lats, alts, band=band)

            lon_shift = lons + d_lon
            lat_shift = lats + d_lat

            pixel_x_shift_x, pixel_y_shift_x = self.lon_lat_alt_to_pixel_x_y(lon_shift, lats, alts, band=band)
            pixel_x_shift_y, pixel_y_shift_y = self.lon_lat_alt_to_pixel_x_y(lons, lat_shift, alts, band=band)

            pxx = (pixel_x_shift_x - pixel_x_estimate) / d_lon
            pxy = (pixel_x_shift_y - pixel_x_estimate) / d_lat
            pyx = (pixel_y_shift_x - pixel_y_estimate) / d_lon
            pyy = (pixel_y_shift_y - pixel_y_estimate) / d_lat

            delta_px = pixel_xs - pixel_x_estimate
            delta_py = pixel_ys - pixel_y_estimate

            delta_lat = (delta_py * pxx - pyx * delta_px) / (pyy * pxx - pxy * pyx)
            delta_lon = (delta_px - delta_lat * pxy) / pxx

            new_lons = lons + delta_lon
            new_lats = lats + delta_lat

            new_pixel_x, new_pixel_y = self.lon_lat_alt_to_pixel_x_y(new_lons, new_lats, alts, band=band)

            lons = new_lons
            lats = new_lats

            x_err = np.abs(pixel_xs - new_pixel_x).max()
            y_err = np.abs(pixel_ys - new_pixel_y).max()

            if x_err <= max_pixel_error and y_err <= max_pixel_error:
                break

        return lons, lats


    def _pixel_x_y_to_lon_lat_ray_caster_native(self,
                                                pixels_x,  # type: ndarray
                                                pixels_y,  # type: ndarray
                                                dem,  # type: AbstractDem
                                                dem_sample_distance,  # type: float
                                                dem_highest_alt=None,  # type: float
                                                dem_lowest_alt=None,  # type: float
                                                band=None,  # type: int
                                                ):  # type: (...) -> (ndarray, ndarray, ndarray)

        # TODO put stuff in here to make sure nx and ny are same size
        # TODO put something here to check that the DEM projection and image projection are the same
        ny = None
        nx = None
        is2d = np.ndim(pixels_x) == 2
        if is2d:
            ny, nx = np.shape(pixels_x)
            pixels_x = image_utils.flatten_image_band(pixels_x)
            pixels_y = image_utils.flatten_image_band(pixels_y)

        n_pixels_to_project = len(pixels_x)

        max_alt = dem_highest_alt
        min_alt = dem_lowest_alt

        if max_alt is None:
            max_alt = dem.get_highest_alt()
        if min_alt is None:
            min_alt = dem.get_lowest_alt()
        alt_range = max_alt - min_alt

        # put the max and min alts at 1 percent above and below the maximum returned by the DEM
        max_alt = max_alt + alt_range * 0.01
        min_alt = min_alt - alt_range * 0.01

        lons_max_alt, lats_max_alt = self.pixel_x_y_alt_to_lon_lat(pixels_x, pixels_y, max_alt, band=band)
        lons_min_alt, lats_min_alt = self.pixel_x_y_alt_to_lon_lat(pixels_x, pixels_y, min_alt, band=band)

        # TODO this operation becomes very expensive at very fine DEM resolutions
        # TODO create implementation for a raster DEM that works faster
        # TODO the time consuming operations are obtaining lon/lats for many points as the DEM resolution becomes finer

        ray_horizontal_lens = np.sqrt(
            np.square(lons_max_alt - lons_min_alt) + np.square(lats_max_alt - lats_min_alt))
        n_steps_per_ray = int(np.ceil(np.max(ray_horizontal_lens) / dem_sample_distance) + 1)

        lons_matrix = np.zeros((n_pixels_to_project, n_steps_per_ray)) + np.linspace(0, 1, n_steps_per_ray)
        lats_matrix = np.zeros((n_pixels_to_project, n_steps_per_ray)) + np.linspace(0, 1, n_steps_per_ray)

        lons_matrix = np.tile((lons_min_alt - lons_max_alt), (n_steps_per_ray, 1)).transpose() * \
                      lons_matrix + np.tile(lons_max_alt, (n_steps_per_ray, 1)).transpose()
        lats_matrix = np.tile((lats_min_alt - lats_max_alt), (n_steps_per_ray, 1)).transpose() * \
                      lats_matrix + np.tile(lats_max_alt, (n_steps_per_ray, 1)).transpose()

        all_elevations = dem.get_elevations(np.array(lons_matrix), np.array(lats_matrix))

        ray = np.linspace(max_alt, min_alt, n_steps_per_ray)
        first_ray_intersect_indices = np.zeros(n_pixels_to_project, dtype=np.int)
        ray_step_indices = list(range(n_steps_per_ray))
        ray_step_indices.reverse()
        for i in ray_step_indices:
            does_ray_intersect = all_elevations[:, i] > ray[i]
            first_ray_intersect_indices[np.where(does_ray_intersect)] = i

        all_pixel_indices = np.arange(0, n_pixels_to_project, dtype=int)

        first_ray_intersect_indices = first_ray_intersect_indices - 1
        second_ray_intersect_indices = first_ray_intersect_indices + 1
        b_rays = ray[first_ray_intersect_indices]
        b_alts = all_elevations[all_pixel_indices, first_ray_intersect_indices]

        m_rays = ray[1] - ray[0]
        m_alts = all_elevations[all_pixel_indices, second_ray_intersect_indices] - b_alts

        xs = (b_alts - b_rays) / (m_rays - m_alts)
        intersected_lons = (lons_matrix[all_pixel_indices, second_ray_intersect_indices] -
                            lons_matrix[all_pixel_indices, first_ray_intersect_indices]) * xs + \
                           lons_matrix[all_pixel_indices, first_ray_intersect_indices]
        intersected_lats = (lats_matrix[all_pixel_indices, second_ray_intersect_indices] -
                            lats_matrix[all_pixel_indices, first_ray_intersect_indices]) * xs + \
                           lats_matrix[all_pixel_indices, first_ray_intersect_indices]
        intersected_alts = (all_elevations[all_pixel_indices, second_ray_intersect_indices] -
                            all_elevations[all_pixel_indices, first_ray_intersect_indices]) * xs + \
                           all_elevations[all_pixel_indices, first_ray_intersect_indices]

        if is2d:
            intersected_lons = image_utils.unflatten_image_band(intersected_lons, nx, ny)
            intersected_lats = image_utils.unflatten_image_band(intersected_lats, nx, ny)
            intersected_alts = image_utils.unflatten_image_band(intersected_alts, nx, ny)

        return intersected_lons, intersected_lats, intersected_alts


    def pixel_x_y_to_lon_lat_alt(self,
                                 pixels_x,  # type: ndarray
                                 pixels_y,  # type: ndarray
                                 dem,  # type: AbstractDem
                                 world_proj=None,  # type: Proj
                                 dem_sample_distance=None,  # type: float
                                 dem_highest_alt=None,  # type: float
                                 dem_lowest_alt=None,  # type: float
                                 band=None,  # type: int
                                 ):  # type: (...) -> (float, float, float)

        DEFAULT_DEM_SAMPLE_DISTANCE = 5
        if dem_sample_distance is None:
            dem_sample_distance = DEFAULT_DEM_SAMPLE_DISTANCE

        if world_proj is None:
            world_proj = self.get_projection()

        native_lons, native_lats, native_alts = self._pixel_x_y_to_lon_lat_ray_caster_native(pixels_x, pixels_y, dem,
                                                                                             dem_sample_distance,
                                                                                             dem_highest_alt,
                                                                                             dem_lowest_alt,
                                                                                             band=band)

        if world_proj.srs != self.get_projection().srs:
            lons, lats = proj_transform(self.get_projection(), world_proj, native_lons, native_lats)
            return lons, lats, native_alts
        else:
            return native_lons, native_lats, native_alts

    def get_projection(self):  # type: (...) -> Proj
        return self._projection

    def set_projection(self,
                       projection  # type: Proj
                       ):  # type: (...) -> None
        self._projection = projection

    def get_approximate_lon_lat_center(self):  # type: (...) -> (float, float)
        return self._lon_lat_center_approximate

    def set_approximate_lon_lat_center(self,
                                       lon,  # type: float
                                       lat  # type: float
                                       ):  # type: (...) -> (float, float)
        self._lon_lat_center_approximate = (lon, lat)

    def bands_coregistered(self):  # type: (...) -> bool
        return self._bands_coregistered
