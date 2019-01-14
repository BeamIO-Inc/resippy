from __future__ import division

from numpy.core.multiarray import ndarray

from resippy.image_objects.earth_overhead.abstract_earth_overhead_point_calc import AbstractEarthOverheadPointCalc
from resippy.photogrammetry.dem.abstract_dem import AbstractDem

import numpy as np
from resippy.utils import file_utils as file_utils
from resippy.utils import string_utils as string_utils
import os


class Pix4dPointCalc(AbstractEarthOverheadPointCalc):

    def __init__(self):
        super(Pix4dPointCalc, self).__init__()
        self.tangential_distortion_1 = None
        self.tangential_distortion_2 = None
        self.radial_distortion_1 = None
        self.radial_distortion_2 = None
        self.radial_distortion_3 = None
        self.distortion_model = None
        self.reverse_x_pixels = True
        self.reverse_y_pixels = True
        self.f_pixels = None
        self.principal_point_x_pixels = None
        self.principal_point_y_pixels = None
        self.r_matrix = None
        self.camera_translation = None
        self.npix_x = None
        self.npix_y = None
        self.image_key = None

    def _pixel_x_y_alt_to_lon_lat_native(self,
                                         pixel_xs,  # type: ndarray
                                         pixel_ys,  # type: ndarray
                                         alts=None,  # type: ndarray
                                         band=None  # type: int
                                         ):  # type: (...) -> (ndarray, ndarray)
        pass

    def _get_reverse_principal_point_x(self):  # type: (...) -> float
        return self.npix_x - self.principal_point_x_pixels

    def _get_reverse_principal_point_y(self):  # type: (...) -> float
        return self.npix_y - self.principal_point_y_pixels

    @classmethod
    def init_from_params(cls,
                         image_key,  # type: str
                         all_params,  # type: dict
                         distortion_model="regular",  # type: str
                         ):  # type: (...) -> Pix4dPointCalc
        point_calc = Pix4dPointCalc()

        image_key_basename = os.path.basename(image_key)
        point_calc.image_key = image_key_basename
        point_calc.r_matrix = all_params[image_key_basename]['camera_rotation_r']
        point_calc.camera_translation = all_params[image_key_basename]['external_params'][0:3]

        pixels_per_mm = all_params[image_key_basename]["n_x_pixels"] / all_params[image_key_basename]["internal_params"]["fpa_width_mm"]
        f_pixels = all_params[image_key_basename]["internal_params"]["F"] * pixels_per_mm
        principal_point_x_pixels = all_params[image_key_basename]["internal_params"]["Px"] * pixels_per_mm
        principal_point_y_pixels = all_params[image_key_basename]["internal_params"]["Py"] * pixels_per_mm

        point_calc.principal_point_x_pixels = principal_point_x_pixels
        point_calc.principal_point_y_pixels = principal_point_y_pixels
        point_calc.npix_x = all_params[image_key_basename]['n_x_pixels']
        point_calc.npix_y = all_params[image_key_basename]['n_y_pixels']
        point_calc.f_pixels = f_pixels
        point_calc.tangential_distortion_1 = all_params[image_key_basename]['tangential_distortion'][0]
        point_calc.tangential_distortion_2 = all_params[image_key_basename]['tangential_distortion'][1]
        point_calc.radial_distortion_1 = all_params[image_key_basename]['radial_distortion'][0]
        point_calc.radial_distortion_2 = all_params[image_key_basename]['radial_distortion'][1]
        point_calc.radial_distortion_3 = all_params[image_key_basename]['radial_distortion'][1]
        point_calc.distortion_model = distortion_model

        approximate_lon = all_params[image_key_basename]['external_params'][0]
        approximate_lat = all_params[image_key_basename]['external_params'][1]

        point_calc.set_approximate_lon_lat_center(approximate_lon, approximate_lat)
        point_calc.set_projection(all_params['projection'])

        return point_calc

    def _lon_lat_alt_to_pixel_x_y_native(self,
                                         lons,  # type: ndarray
                                         lats,  # type: ndarray
                                         alts,  # type: ndarray
                                         band=None  # type: int
                                         ):  # type: (...) -> (ndarray, ndarray)
        pixel_locs = self.world_xyzs_to_pixel_locations(lons, lats, alts,
                                                        distortion_model=self.distortion_model,
                                                        reverse_y_pixels=self.reverse_y_pixels,
                                                        reverse_x_pixels=self.reverse_x_pixels)
        return pixel_locs

    def pixel_x_y_to_lon_lat_alt(self,
                                 pixel_x,  # type: ndarray
                                 pixel_y,  # type: ndarray
                                 dem=None,  # type: AbstractDem
                                 band=None  # type: int
                                 ):  # type: (...) -> (ndarray, ndarray, ndarray)
        pass

    def world_xyzs_to_pixel_locations(self,
                                      world_x_arr,  # type: ndarray
                                      world_y_arr,  # type: ndarray
                                      world_z_arr,  # type: ndarray
                                      distortion_model=None,  # type: str
                                      reverse_y_pixels=True,  # type: bool
                                      reverse_x_pixels=True  # type: bool
                                      ):  # type: (...) -> ndarray

        local_world_xyz_arr = np.zeros((3, len(world_x_arr)))
        local_world_xyz_arr[0, :] = world_x_arr
        local_world_xyz_arr[1, :] = world_y_arr
        local_world_xyz_arr[2, :] = world_z_arr

        camera_coords = self.world_to_camera_coordinates(local_world_xyz_arr)
        pixel_coords = None
        if distortion_model is None:
            pixel_coords = np.array(
                self.camera_coords_to_pixel_coords_no_distortion(camera_coords))
        if distortion_model == "regular":
            pixel_coords = np.array(self.camera_coords_to_pixel_coords_with_distortion(camera_coords))

        if reverse_y_pixels:
            pixel_coords[1] = self.npix_y - pixel_coords[1]
        if reverse_x_pixels:
            pixel_coords[0] = self.npix_x - pixel_coords[0]

        return pixel_coords

    def read_calibrated_external_camera_parameters(self,
                                                   calibrated_external_params_file  # type: str
                                                   ):  # type: (...) -> dict
        external_params = {}
        with open(calibrated_external_params_file, 'r') as f:
            lines = f.readlines()
        cleaned_up_lines = [string_utils.remove_newlines(x) for x in lines][1:]
        for file_params in cleaned_up_lines:
            fname, x, y, z, omega, phi, kappa = file_params.split(" ")
            external_params[fname] = [float(x), float(y), float(z), float(omega), float(phi), float(kappa)]
        return external_params

    def read_pix4d_calibrated_internal_camera_parameters(self,
                                                         internal_params_cam_file,  # type: str
                                                         search_text="Pix4D"  # type: str
                                                         ):  # type: (...) -> list
        with open(internal_params_cam_file, 'r') as f:
            lines = f.readlines()
        lines = [string_utils.remove_newlines(x) for x in lines]
        new_camera_param_start_locations = []

        for i, line in enumerate(lines):
            if search_text in line:
                new_camera_param_start_locations.append(i)

        new_camera_param_end_locations = new_camera_param_start_locations[1:]
        new_camera_param_end_locations.append(i)

        camera_params_list = []

        for i in range(len(new_camera_param_start_locations)):
            internal_params_dict = {}
            text_block = lines[new_camera_param_start_locations[i] + 1:new_camera_param_end_locations[i]]
            text_block = list(filter(lambda a: a != '', text_block))
            for txt in text_block:
                if txt.startswith("#"):
                    if "Focal Length mm" in txt:
                        fpa_width_mm, fpa_height_mm = txt[txt.rfind(" ") + 1:].replace("mm", "").split("x")
                        internal_params_dict["fpa_width_mm"] = float(fpa_width_mm)
                        internal_params_dict["fpa_height_mm"] = float(fpa_height_mm)
                    else:
                        pass
                else:
                    key, val = txt.split(" ")
                    internal_params_dict[key] = float(val)
            camera_params_list.append(internal_params_dict)
        return camera_params_list

    def world_to_camera_coordinates(self,
                                    local_world_xyzs  # type: ndarray
                                    ):  # type: (...) -> ndarray
        translated_xyzs = np.zeros(np.shape(local_world_xyzs))
        translated_xyzs[0, :] = local_world_xyzs[0, :] - self.camera_translation[0]
        translated_xyzs[1, :] = local_world_xyzs[1, :] - self.camera_translation[1]
        translated_xyzs[2, :] = local_world_xyzs[2, :] - self.camera_translation[2]

        camera_coords = np.matmul(self.r_matrix, translated_xyzs)
        return camera_coords

    def camera_coords_to_pixel_coords_with_distortion(self,
                                                      camera_coords_xyz  # type: ndarray
                                                      ):  # type: (...) -> (float, float)
        principal_point_x_pixels = self.principal_point_x_pixels
        principal_point_y_pixels = self.principal_point_y_pixels
        if self.reverse_x_pixels:
            principal_point_x_pixels = self._get_reverse_principal_point_x()
        if self.reverse_y_pixels:
            principal_point_y_pixels = self._get_reverse_principal_point_y()

        x_h = camera_coords_xyz[0, :] / camera_coords_xyz[2, :]
        y_h = camera_coords_xyz[1, :] / camera_coords_xyz[2, :]

        r_squared = np.power(x_h, 2) + np.power(y_h, 2)
        r = np.power(r_squared, 0.5)

        x_hd = (1 + self.radial_distortion_1 * r_squared + self.radial_distortion_2 * np.power(r, 4) + self.radial_distortion_3 * np.power(r, 6)) * x_h + 2 * self.tangential_distortion_1 * x_h * y_h + self.tangential_distortion_2 * (np.power(r, 2) + 2 * np.power(x_h, 2))
        y_hd = (1 + self.radial_distortion_1 * r_squared + self.radial_distortion_2 * np.power(r, 4) + self.radial_distortion_3 * np.power(r, 6)) * y_h + 2 * self.tangential_distortion_2 * x_h * y_h + self.tangential_distortion_1 * (np.power(r, 2) + 2 * np.power(y_h, 2))

        x_d = -self.f_pixels * x_hd + principal_point_x_pixels
        y_d = -self.f_pixels * y_hd + principal_point_y_pixels

        return x_d, y_d

    def camera_coords_to_pixel_coords_no_distortion(self,
                                                    camera_coords_xyz  # type: ndarray
                                                    ):  # type: (...) -> (float, float)

        principal_point_x = self.principal_point_x_pixels
        principal_point_y = self.principal_point_y_pixels

        if self.reverse_x_pixels:
            principal_point_x = self._get_reverse_principal_point_x()
        if self.reverse_y_pixels:
            principal_point_y = self._get_reverse_principal_point_y()

        x_u = -self.f_pixels * camera_coords_xyz[0, :] / camera_coords_xyz[2, :] + principal_point_x
        y_u = -self.f_pixels * camera_coords_xyz[1, :] / camera_coords_xyz[2, :] + principal_point_y
        return x_u, y_u
