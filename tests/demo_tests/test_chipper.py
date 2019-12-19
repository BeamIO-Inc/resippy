import unittest
from resippy.test_runner import demo_data_base_dir
from resippy.utils import file_utils as file_utils
from resippy.utils.image_utils import image_chipper
from resippy.utils.image_utils import image_utils
from imageio import imread
import numpy as np
import random


class TestChipper(unittest.TestCase):
    print("TEST CHIPPER TESTS")

    def setUp(self):
        print("")
        self.png_fullpath = file_utils.get_path_from_subdirs(demo_data_base_dir,
                                                             ["image_data",
                                                              "overhead_vehicle_data",
                                                              "Potsdam_ISPRS",
                                                              "top_potsdam_2_14_RGB.png"])
        self.png_image_data = imread(self.png_fullpath)
        png_shape = self.png_image_data.shape
        self.ny = png_shape[0]
        self.nx = png_shape[1]
        self.nbands = png_shape[2]

    def test_overlap(self):
        print("")
        chip_size = 256
        npix_overlap = 128
        rgb_chips, rgb_chips_indices = image_chipper.chip_entire_image_to_memory(self.png_image_data,
                                                                                 bands=[0, 1, 2],
                                                                                 chip_nx_pixels=chip_size,
                                                                                 chip_ny_pixels=chip_size)
        rgb_y_overlap_chips, rgb_y_overlap_chip_indices = image_chipper.\
            chip_entire_image_to_memory(self.png_image_data,
                                        npix_overlap_y=npix_overlap,
                                        chip_nx_pixels=chip_size,
                                        chip_ny_pixels=chip_size,
                                        bands=[0, 1, 2])
        n_rgb_chips = len(rgb_chips)
        n_rgb_w_overlap = len(rgb_y_overlap_chip_indices)
        ny, nx, nbands = image_utils.get_image_ny_nx_nbands(self.png_image_data)

        assert np.ceil(ny/chip_size) * np.ceil(nx/chip_size) == n_rgb_chips
        assert np.ceil((ny/chip_size) * 2 - 1) * np.ceil(nx/chip_size) == n_rgb_w_overlap
        print("overlap is working as expected")
        print("OVERLAP TEST PASSED")

    def test_bands(self):
        print("")
        rgb_bands_list = [0, 1, 2]
        full_chips, full_chips_indices = image_chipper.chip_entire_image_to_memory(self.png_image_data)
        rgb_chips, rgb_chips_indices = image_chipper.chip_entire_image_to_memory(self.png_image_data,
                                                                                 bands=rgb_bands_list)
        n_chips_full = full_chips.shape[0]
        n_chips_rgb = rgb_chips.shape[0]
        assert n_chips_full == n_chips_rgb
        print("number of full chips and number of grayscale chips match")

        for i in range(n_chips_full):
            full_chip = full_chips[i]
            rgb_chip = rgb_chips[i]
            assert(full_chip[:, :, rgb_bands_list] == rgb_chip).all()
        print("full chips using bands subset list matches rgb chips")
        print("TEST BANDS TEST PASSED")

    def test_grayscale(self):
        print("")
        band_num = 0
        full_chips, full_chips_indices = image_chipper.chip_entire_image_to_memory(self.png_image_data)
        grayscale_chips, grayscale_chip_indices = image_chipper.chip_entire_image_to_memory(self.png_image_data,
                                                                                            bands=band_num)

        n_chips_full = full_chips.shape[0]
        n_chips_grayscale = grayscale_chips.shape[0]
        assert n_chips_full == n_chips_grayscale
        print("test chipper - number of full chips and number of rgb chips match")

        for i in range(n_chips_full):
            full_chip = full_chips[i]
            grayscale_chip = grayscale_chips[i]
            assert grayscale_chip.shape != full_chip.shape
            assert (full_chip[:, :, band_num] == grayscale_chip).all()
        print("test chipper - grayscale test passed")

    def test_lower_right(self):
        print("")
        chip_nx_pixels = 217
        chip_ny_pixels = 217
        full_chips, full_chips_indices = image_chipper.chip_entire_image_to_memory(self.png_image_data,
                                                                                   chip_ny_pixels=chip_ny_pixels,
                                                                                   chip_nx_pixels=chip_nx_pixels)
        lower_right_image = self.png_image_data[(self.ny-chip_ny_pixels):, (self.nx-chip_nx_pixels):, :]
        assert (full_chips[-1] == lower_right_image).any()
        print("last chip matches the lower corner of the original image.")
        print("LOWER RIGHT TEST PASSED")

    def test_keep_within_bounds(self):
        print("")
        ul_y = [self.ny - 10, -10]
        ul_x = [self.nx - 10, -10]
        chip_ny_pixels = 217
        chip_nx_pixels = 217
        chips, ul_indices = image_chipper.chip_images_by_pixel_upper_lefts(self.png_image_data, ul_y, ul_x,
                                                                           chip_ny_pixels=chip_ny_pixels,
                                                                           chip_nx_pixels=chip_nx_pixels)
        lower_right_image = self.png_image_data[(self.ny - chip_ny_pixels):, (self.nx - chip_nx_pixels):, :]
        assert(chips[0] == lower_right_image).all()

        upper_left_image = self.png_image_data[0:chip_ny_pixels, 0:chip_nx_pixels]
        assert(chips[1] == upper_left_image).all()

        print("chipping out within image bounds")
        print("")

    def test_chip_by_centers(self):
        n_samples = 20
        chip_size = 256
        y_uls = random.sample(list(np.arange(0, self.ny)), n_samples)
        x_uls = random.sample(list(np.arange(0, self.nx)), n_samples)

        x_centers = np.add(x_uls, chip_size/2).astype(int)
        y_centers = np.add(y_uls, chip_size/2).astype(int)

        chips_by_ul, indices_by_ul = image_chipper.chip_images_by_pixel_upper_lefts(
            self.png_image_data, pixel_y_ul_list=y_uls, pixel_x_ul_list=x_uls,
            chip_nx_pixels=chip_size, chip_ny_pixels=chip_size)

        chips_by_centers, indices_by_ul = image_chipper.chip_images_by_pixel_centers(
            self.png_image_data, pixel_y_center_list=y_centers, pixel_x_center_list=x_centers,
            chip_nx_pixels=chip_size, chip_ny_pixels=chip_size)

        assert(chips_by_ul == chips_by_centers).all()
        print("chipping by centers matches chipping by ul")
        print("")


if __name__ == '__main__':
    unittest.main()
