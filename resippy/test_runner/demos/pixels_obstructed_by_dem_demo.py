from __future__ import division

from resippy.test_runner import demo_data_base_dir, demo_data_save_dir
from resippy.utils import file_utils
import resippy.utils.pix4d_utils as pix4d
from resippy.photogrammetry.dem.dem_factory import DemFactory
from resippy.image_objects.image_factory import ImageFactory
import resippy.photogrammetry.ortho_tools as ortho_tools
from resippy.utils.image_utils import image_utils
import imageio

import os

import time

save_dir = os.path.join(demo_data_save_dir, "pixels_obstructed_by_dem_demo")
file_utils.make_dir_if_not_exists(save_dir)

micasense_dir = file_utils.get_path_from_subdirs(demo_data_base_dir, ['image_data',
                                                                      'multispectral',
                                                                      'micasense',
                                                                      '20181019_hana',
                                                                      '1703',
                                                                      'micasense',
                                                                      'processed'
                                                                      ])


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


def pixels_obstructed_by_dem_demo():
    band = 0
    npix_x = 1000
    npix_y = 1000
    dem_resolution = 5
    extent = ortho_tools.get_extent(micasense_image, dem, bands=band).bounds
    min_x = extent[0]
    min_y = extent[1]
    max_x = extent[2]
    max_y = extent[3]
    ground_grid = ortho_tools.create_ground_grid(min_x, max_x, min_y, max_y, npix_x, npix_y)
    lons = ground_grid[0]
    lats = ground_grid[1]
    alts = dem.get_elevations(lons, lats)
    pixels_x, pixels_y = micasense_image.get_point_calculator().lon_lat_alt_to_pixel_x_y(lons, lats, alts, band=band)
    tic = time.process_time()
    obstructed_pixels = ortho_tools.are_pixels_obstructed_by_dem(micasense_image, pixels_x, pixels_y, lons, lats, dem,
                                                                 dem_resolution=dem_resolution)
    toc = time.process_time()
    print("took " + str(toc - tic) + " seconds to find pixels obstructed by the dem")

    obstructed_pixels_colormapped = image_utils.apply_colormap_to_grayscale_image(obstructed_pixels)
    imageio.imwrite(os.path.join(save_dir, "pixels_obstructed_by_dem.png"), obstructed_pixels_colormapped)

    corresponding_dem = dem.get_elevations(lons, lats)
    dem_colormapped = image_utils.apply_colormap_to_grayscale_image(corresponding_dem)
    imageio.imwrite(os.path.join(save_dir, "dem.png"), dem_colormapped)


def main():
    pixels_obstructed_by_dem_demo()


if __name__ == '__main__':
    main()
