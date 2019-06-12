import resippy.utils.pix4d_utils as pix4d
import resippy.utils.photogrammetry_utils as photogrammetry_utils
import resippy.photogrammetry.crs_defs as crs_defs
import resippy.photogrammetry.ortho_tools as ortho_tools
from resippy.photogrammetry.dem.dem_factory import DemFactory
from resippy.image_objects.image_factory import ImageFactory
from resippy.test_runner import demo_data_base_dir, demo_data_save_dir
from resippy.utils import file_utils as file_utils

import numpy as np
import os


# choose a gsd
gsd_m = 0.5
# constants calculated from https://msi.nga.mil/MSISiteContent/StaticFiles/Calculators/degree.html
# based on approx latitude of center of images
gsd_lon = gsd_m / 81417.0
gsd_lat = gsd_m / 111095.0

# set up filename bundle for a frame, note that it is assumed that the band number is the last item in the filename

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

# set up where to save output orthos
save_dir = os.path.join(demo_data_save_dir, "micasense_ortho_demos")
file_utils.make_dir_if_not_exists(save_dir)

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
dem_image = DemFactory.from_gtiff_file(dsm_fullpath)
dem_native_projection = dem_image.get_projection()

if dem_native_projection.srs != micasense_native_projection.srs:
    raise ValueError("pix4d dem and micasense projections should be in same CRS!")

# get the extent in native coordinates
extent_native = ortho_tools.get_extent(micasense_image, dem=dem_image, pixel_error_threshold=0.1)


# functions to make various ortho types
def ortho_micasense_to_dem_native():
    output_fullpath = os.path.join(save_dir, "micasense_ortho_w_dem_native.tif")

    ortho_nx, ortho_ny = photogrammetry_utils.get_nx_ny_pixels_in_extent(extent_native, gsd_m, gsd_m)
    print("making ortho of size: " + str(ortho_nx) + "x" + str(ortho_ny))
    ortho_tools.create_ortho_gtiff_image_world_to_sensor(micasense_image, ortho_nx, ortho_ny, extent_native,
                                                         world_proj=dem_image.get_projection(),
                                                         dem=dem_image, nodata_val=0,
                                                         output_fname=output_fullpath)
    print("wrote orthorectified micasense image to " + output_fullpath)


def ortho_micasense_to_dem_4326():
    extent_4326 = photogrammetry_utils.reproject_geometry(extent_native, dem_native_projection, crs_defs.PROJ_4326)
    ortho_nx, ortho_ny = photogrammetry_utils.get_nx_ny_pixels_in_extent(extent_4326, gsd_lon, gsd_lat)
    print("making ortho of size: " + str(ortho_nx) + "x" + str(ortho_ny))

    output_fullpath = os.path.join(save_dir, "micasense_ortho_w_dem_4326.tif")
    ortho_tools.create_ortho_gtiff_image_world_to_sensor(micasense_image, ortho_nx, ortho_ny, extent_4326,
                                                         world_proj=crs_defs.PROJ_4326, dem=dem_image, nodata_val=0,
                                                         output_fname=output_fullpath)
    print("wrote orthorectified micasense image to " + output_fullpath)


def ortho_micasense_to_flat_earth_native():
    elevation = np.nanmean(dem_image.dem_data)
    const_elevation_dem = DemFactory.constant_elevation(elevation)

    ortho_nx, ortho_ny = photogrammetry_utils.get_nx_ny_pixels_in_extent(extent_native, gsd_m, gsd_m)
    print("making ortho of size: " + str(ortho_nx) + "x" + str(ortho_ny))

    output_fullpath = os.path.join(save_dir, "micasense_ortho_w_const_elevation_dem_native.tif")

    ortho_tools.create_ortho_gtiff_image_world_to_sensor(micasense_image, ortho_nx, ortho_ny, extent_native,
                                                         world_proj=dem_image.get_projection(),
                                                         dem=const_elevation_dem, nodata_val=0,
                                                         output_fname=output_fullpath)
    print("wrote orthorectified micasense image to " + output_fullpath)


def ortho_micasense_to_flat_earth_4326():
    extent_4326 = photogrammetry_utils.reproject_geometry(extent_native, dem_native_projection, crs_defs.PROJ_4326)
    elevation = np.nanmean(dem_image.dem_data)
    const_elevation_dem = DemFactory.constant_elevation(elevation)

    ortho_nx, ortho_ny = photogrammetry_utils.get_nx_ny_pixels_in_extent(extent_4326, gsd_lon, gsd_lat)
    print("making ortho of size: " + str(ortho_nx) + "x" + str(ortho_ny))
    output_fullpath = os.path.join(save_dir, "micasense_ortho_w_const_elevation_dem_4326.tif")

    ortho_tools.create_ortho_gtiff_image_world_to_sensor(micasense_image, ortho_nx, ortho_ny, extent_4326,
                                                         world_proj=crs_defs.PROJ_4326,
                                                         dem=const_elevation_dem, nodata_val=0,
                                                         output_fname=output_fullpath)
    print("wrote orthorectified micasense image to " + output_fullpath)


def ortho_micasense_to_flat_earth_3857():
    extent_3857 = photogrammetry_utils.reproject_geometry(extent_native, dem_native_projection, crs_defs.PROJ_3857)
    elevation = np.nanmean(dem_image.dem_data)

    ortho_nx, ortho_ny = photogrammetry_utils.get_nx_ny_pixels_in_extent(extent_3857, gsd_m, gsd_m)
    print("making ortho of size: " + str(ortho_nx) + "x" + str(ortho_ny))

    const_elevation_dem = DemFactory.constant_elevation(elevation)
    output_fullpath = os.path.join(save_dir, "micasense_ortho_w_const_elevation_dem_3857.tif")

    ortho_tools.create_ortho_gtiff_image_world_to_sensor(micasense_image, ortho_nx, ortho_ny, extent_3857,
                                                         world_proj=crs_defs.PROJ_3857,
                                                         dem=const_elevation_dem, nodata_val=0,
                                                         output_fname=output_fullpath)
    print("wrote orthorectified micasense image to " + output_fullpath)


def main():
    ortho_micasense_to_dem_native()
    ortho_micasense_to_dem_4326()
    ortho_micasense_to_flat_earth_native()
    ortho_micasense_to_flat_earth_4326()
    ortho_micasense_to_flat_earth_3857()


if __name__ == '__main__':
    main()

