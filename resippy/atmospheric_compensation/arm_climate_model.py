import numpy
from numpy import ndarray
from PIL import Image
from resippy.utils import coordinate_conversions
from resippy.utils.image_utils import image_utils
from resippy.atmospheric_compensation.utils import hemisphere_coordinate_conversions
from scipy.ndimage import map_coordinates

import matplotlib.pyplot as plt


class ArmClimateModel:
    def __init__(self):
        self._cloud_mask = None  # type: ndarray
        self._arm_center_xyz = None  # type: (float, float, float)
        self._uv_image = None  # type: ndarray
        self._default_uv_image_size = 2048

    @classmethod
    def from_image_file(cls, image_path):
        image = numpy.array(Image.open(image_path))
        ny, nx, nbands = image.shape
        horizontal_cross_section = image[int(ny / 2), :][:, 0]
        horizontal_start_index = numpy.argmax(horizontal_cross_section < 255)
        horizontal_tmp = horizontal_cross_section[::-1]
        horizontal_end_index = len(horizontal_tmp) - numpy.argmax(horizontal_tmp < 255) - 1
        image_subsection = image[:, horizontal_start_index:horizontal_end_index]

        vertical_cross_section = image_subsection[:, 0][:, 0]
        vertical_start_index = numpy.argmax(vertical_cross_section < 255)
        vertical_tmp = vertical_cross_section[::-1]
        vertical_end_index = len(vertical_tmp) - numpy.argmax(vertical_tmp < 255) - 1
        image_subsection = image_subsection[vertical_start_index: vertical_end_index, :]

        return cls.from_numpy_array(image_subsection)

    # TODO: improve azimuth functionality
    @classmethod
    def from_numpy_array(cls,
                         array,  # type: ndarray
                         n_uv_pixels=None,  # type: int
                         ):
        uv_shape = numpy.shape(array)
        ny = uv_shape[0]
        nx = uv_shape[1]

        # TODO: do stuff here to center image, create accurate UV image map.  For now this does nothing and
        # TODO: we just resize the image to create a UV image
        center_y, center_x = ny / 2, nx / 2
        y_pixel_locs = numpy.repeat(numpy.arange(0, ny), nx).reshape(ny, nx)
        x_pixel_locs = numpy.tile(numpy.arange(0, nx), ny).reshape(ny, nx)
        x_pixel_distances = x_pixel_locs - center_x
        y_pixel_distances = y_pixel_locs - center_y
        zero_elevation_distance = numpy.max((nx, ny)) / 2
        pixel_distances_from_center = numpy.sqrt(numpy.power(x_pixel_distances, 2) + numpy.power(y_pixel_distances, 2))

        m = (numpy.pi / 2) / (-zero_elevation_distance)
        b = numpy.pi / 2

        pixel_elevations = m * pixel_distances_from_center + b
        pixel_elevations[numpy.where(pixel_elevations <= 0)] = 0

        arm_model = cls()
        arm_model._cloud_mask = array
        if n_uv_pixels is None:
            n_uv_pixels = arm_model._default_uv_image_size
        arm_model.uv_image = image_utils.resize_image(array, n_uv_pixels, n_uv_pixels)
        return arm_model

    @property
    def n_uv_pixels(self):
        return self.uv_image.shape[0]

    @property
    def uv_image(self):
        return self._uv_image

    @uv_image.setter
    def uv_image(self, val):
        self._uv_image = val

    @property
    def center_xyz_location(self):
        return self._arm_center_xyz

    @center_xyz_location.setter
    def center_xyz_location(self,
                            val,  # type: (float, float, float)
                            ):
        self._arm_center_xyz = val

    def project_uv_image_pixels_to_cloud_deck(self,
                                              cloud_deck_height,
                                              ):
        y_arr, x_arr = numpy.mgrid[0:self.n_uv_pixels, 0:self.n_uv_pixels]
        az, el = hemisphere_coordinate_conversions.uv_pixel_yx_coords_to_az_el(self.n_uv_pixels, y_arr, x_arr)
        cloud_x, cloud_y, cloud_z = coordinate_conversions.az_el_to_xy_plane(az,
                                                                             el,
                                                                             cloud_deck_height,
                                                                             self.center_xyz_location)
        return cloud_x, cloud_y, cloud_z

    @classmethod
    def translated_arm_model(cls,
                             original_model,  # type: ArmClimateModel
                             cloud_deck_height,  # type: float
                             translated_xyz,  # type: (float, float, float)
                             ):
        x_0, y_0, z_0 = original_model.project_uv_image_pixels_to_cloud_deck(cloud_deck_height)
        x_1 = x_0 - translated_xyz[0]
        y_1 = y_0 - translated_xyz[1]
        z_1 = z_0 - translated_xyz[2]
        new_az, new_el, new_r = coordinate_conversions.xyz_to_az_el_radius(x_1, y_1, z_1)
        new_uv_ypix, new_uv_xpix = \
            hemisphere_coordinate_conversions.az_el_to_uv_pixel_yx_coords(original_model.n_uv_pixels, new_az, new_el)

        warped_uv_image = numpy.zeros_like(original_model.uv_image)
        uv_image_shape = original_model.uv_image.shape
        if len(uv_image_shape) == 3:
            band_1 = map_coordinates(original_model.uv_image[:, :, 0],
                                     [image_utils.flatten_image_band(new_uv_ypix),
                                      image_utils.flatten_image_band(new_uv_xpix)])
            band_1 = numpy.reshape(band_1,
                                   (original_model.n_uv_pixels, original_model.n_uv_pixels))

            band_2 = map_coordinates(original_model.uv_image[:, :, 1],
                                     [image_utils.flatten_image_band(new_uv_ypix),
                                      image_utils.flatten_image_band(new_uv_xpix)])
            band_2 = numpy.reshape(band_2,
                                   (original_model.n_uv_pixels, original_model.n_uv_pixels))

            band_3 = map_coordinates(original_model.uv_image[:, :, 2],
                                     [image_utils.flatten_image_band(new_uv_ypix),
                                      image_utils.flatten_image_band(new_uv_xpix)])
            band_3 = numpy.reshape(band_3,
                                   (original_model.n_uv_pixels, original_model.n_uv_pixels))

            warped_uv_image[:, :, 0] = band_1
            warped_uv_image[:, :, 1] = band_2
            warped_uv_image[:, :, 2] = band_3
        else:
            band_1 = map_coordinates(original_model.uv_image,
                                     [image_utils.flatten_image_band(new_uv_ypix),
                                      image_utils.flatten_image_band(new_uv_xpix)])
            warped_uv_image = numpy.reshape(band_1,
                                            (original_model.n_uv_pixels, original_model.n_uv_pixels))

        new_model = cls.from_numpy_array(warped_uv_image, original_model.n_uv_pixels)
        return new_model
