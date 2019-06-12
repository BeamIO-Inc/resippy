from __future__ import division
import unittest

import resippy.utils.pix4d_utils as pix4d
from resippy.photogrammetry.dem.dem_factory import DemFactory
from resippy.image_objects.image_factory import ImageFactory
from resippy.test_runner import demo_data_base_dir
import resippy.utils.image_utils.image_utils as image_utils
import resippy.utils.file_utils as file_utils
import numpy as np
import os


micasense_dir = file_utils.get_path_from_subdirs(demo_data_base_dir, ['image_data',
                                                                      'multispectral',
                                                                      'micasense',
                                                                      '20181019_hana',
                                                                      '1703',
                                                                      'micasense',
                                                                      'processed'
                                                                      ])


# set up filename bundle for a frame, note that it is assumed that the band number is the last item in the filename
band_fname_dict = {}
for cam_num in range(1, 6):
    band_num = str(cam_num)
    band_fname_dict['band' + band_num] = \
        file_utils.get_path_from_subdirs(micasense_dir, ['merged', 'L0_IMG_0404_' + band_num + '.tif'])


# set up path to pix4d parameters used to build point calculator etc
params_dir = file_utils.get_path_from_subdirs(micasense_dir, ['pix4d_1703_mica_imgs85_1607',
                                                              '1_initial',
                                                              'params'])


# set up where digital surface model is stored
dsm_fullpath = file_utils.get_path_from_subdirs(micasense_dir,
                                                ['pix4d_1703_mica_imgs85_1607',
                                                 '3_dsm_ortho',
                                                 'pix4d_1704_mica_85_1607_dsm.tif'])



# load up the pix4d info
pix4d_master_dict = pix4d.make_master_dict(params_dir)

# make the micasense camera object
micasense_image = ImageFactory.micasense.from_image_number_and_pix4d(band_fname_dict, pix4d_master_dict)
micasense_native_projection = micasense_image.get_point_calculator().get_projection()

# make the dem camera object
dem = DemFactory.from_gtiff_file(dsm_fullpath)
dem.set_interpolation_to_bilinear()
dem.set_interpolation_to_nearest()
dem_native_projection = dem.get_projection()

if dem_native_projection.srs != micasense_native_projection.srs:
    raise ValueError("pix4d dem and micasense projections should be in same CRS!")


class TestRayCaster(unittest.TestCase):

    def test_micasense_pixel_to_world(self):

        pixel_sf = 0.1
        dem_resolution = 0.05
        band = 0

        n_decimals = 4
        print("MICASENSE PIXEL TO WORLD TEST")
        print("THIS TEST ASSUMES THAT THE INTERPOLATED HIGHEST ALTITUDE RETRIEVED FROM RAY CASTING IS CORRECT.")

        pixels_x, pixels_y = image_utils.create_pixel_grid(micasense_image.get_metadata().get_npix_x(),
                                                           micasense_image.get_metadata().get_npix_y(),
                                                           scale_factor=pixel_sf)

        lons, lats, alts = micasense_image.get_point_calculator(). \
            pixel_x_y_to_lon_lat_alt(pixels_x, pixels_y, dem, dem_sample_distance=dem_resolution, band=band)
        casted_pixels_x, casted_pixels_y = micasense_image.get_point_calculator().lon_lat_alt_to_pixel_x_y(lons, lats, alts, band=band)

        np.testing.assert_almost_equal(pixels_x, casted_pixels_x, decimal=n_decimals)
        np.testing.assert_almost_equal(pixels_y, casted_pixels_y, decimal=n_decimals)

        print("pixels projected onto the earth and back agree to within " + str(n_decimals) + " decimal places")

    def test_micasense_pixel_to_world_2(self):
        pixel_sf = 0.25
        dem_resolution = 1
        band = 0

        n_decimals = 4
        print("MICASENSE PIXEL TO WORLD TEST 2")
        print("TIMINGS AT HALF METER DEM RESOLUTION")

        pixels_x, pixels_y = image_utils.create_pixel_grid(micasense_image.get_metadata().get_npix_x(),
                                                           micasense_image.get_metadata().get_npix_y(),
                                                           scale_factor=pixel_sf)

        lons, lats, alts = micasense_image.get_point_calculator(). \
            pixel_x_y_to_lon_lat_alt(pixels_x, pixels_y, dem, dem_sample_distance=dem_resolution, band=band)
        casted_pixels_x, casted_pixels_y = micasense_image.get_point_calculator().lon_lat_alt_to_pixel_x_y(lons, lats,
                                                                                                           alts,
                                                                                                           band=band)

        np.testing.assert_almost_equal(pixels_x, casted_pixels_x, decimal=n_decimals)
        np.testing.assert_almost_equal(pixels_y, casted_pixels_y, decimal=n_decimals)

        print("pixels projected onto the earth and back agree to within " + str(n_decimals) + " decimal places")


if __name__ == '__main__':
    unittest.main()





