from resippy.utils.envi_utils import read_envi_header
from resippy.utils.envi_utils import read_envi_image
from resippy.image_objects.envi.envi_image_factory import EnviImageFactory
from resippy.image_objects.image_factory import ImageFactory
from resippy.image_objects.earth_overhead.igm.igm_image import IgmImage
from resippy.photogrammetry import crs_defs
from resippy.photogrammetry import ortho_tools
from resippy.photogrammetry.dem.constant_elevation_dem import ConstantElevationDem
import os
import matplotlib.pyplot as plt

import numpy

raw_dir = os.path.expanduser("~/Data/hyperspec3/raw")
ortho_dir = os.path.expanduser("~/Data/hyperspec3/ortho")
output_dir = os.path.expanduser("~/Downloads/")

envi_ortho_header_fname = os.path.join(ortho_dir, "raw_1440_0-0-0-0_or.hdr")
envi_ortho_fname = os.path.join(ortho_dir, "raw_1440_0-0-0-0_or")

envi_raw_header_fname = os.path.join(raw_dir, "raw_1440.hdr")
envi_raw_fname = os.path.join(raw_dir, "raw_1440")

envi_igm_header_fname = os.path.join(raw_dir, "raw_1440_0-0-0-0_igm.hdr")
envi_igm_fname = os.path.join(raw_dir, "raw_1440_0-0-0-0_igm")


raw_image_object = EnviImageFactory.from_file(envi_raw_fname, envi_raw_header_fname)

red_band_num = 80
green_band_num = 40
blue_band_num = 7

r = raw_image_object.read_band_from_disk(red_band_num)
g = raw_image_object.read_band_from_disk(green_band_num)
b = raw_image_object.read_band_from_disk(blue_band_num)

ny, nx = r.shape

raw_rgb = numpy.zeros((ny, nx, 3))
raw_rgb[:, :, 0] = r
raw_rgb[:, :, 1] = g
raw_rgb[:, :, 2] = b

raw_rgb_8_bit = raw_rgb / 4096.0 * 255
raw_rgb_8_bit = numpy.asarray(raw_rgb_8_bit, int)

igm_data = ImageFactory.envi.from_file(envi_igm_fname)

igm_lons = igm_data.read_band_from_disk(0)
igm_lats = igm_data.read_band_from_disk(1)
igm_alts = igm_data.read_band_from_disk(2)

igm_image_obj = IgmImage.from_params(raw_rgb_8_bit, igm_lons, igm_lats, igm_alts, 3, crs_defs.PROJ_4326)

# TODO: here we would do keypoint matching on the ortho'd image, then get the lat/lon coordinates for the keypoint matches, and do the step below to get the pixel coordinates from the raw frames.

# found inspecting the image in qgis and selecting a point on the center of a car in the parking lot.
test_car_coord_lon = -71.6003424
test_car_coord_lat = 42.4285518

dem = ConstantElevationDem()

envi_ortho_image_data = read_envi_image(envi_ortho_fname, envi_ortho_header_fname)
envi_ortho_header = read_envi_header(envi_ortho_header_fname)
mapinfo_fields = envi_ortho_header['map info'].split(',')
lon_ul = float(mapinfo_fields[3])
lat_ul = float(mapinfo_fields[4])
lon_gsd = float(mapinfo_fields[5])
lat_gsd = float(mapinfo_fields[6])

geot = (lon_ul, lon_gsd, 0, lat_ul, 0, -lat_gsd)
geot_image = ImageFactory.geotiff.from_numpy_array(envi_ortho_image_data, geot, crs_defs.PROJ_4326)

# taken by viewing the ortho'd image using matplotlib and selecting a pixel on the center of a car in the parking lot
geot_pixel_x = 311
geot_pixel_y = 780

lon_lat_from_ortho = geot_image.pointcalc.pixel_x_y_alt_to_lon_lat(geot_pixel_x, geot_pixel_y, 0, crs_defs.PROJ_4326)

raw_x_y_pixel_coord_from_known_lon_lat = igm_image_obj.point_calc.lon_lat_alt_to_pixel_x_y(test_car_coord_lon, test_car_coord_lat, 0)
reprojected_test_car_lon_lat_coord = igm_image_obj.point_calc.pixel_x_y_to_lon_lat_alt(int(raw_x_y_pixel_coord_from_known_lon_lat[0]), int(raw_x_y_pixel_coord_from_known_lon_lat[1]), dem)

raw_x_y_pixel_coord_from_geotiff_kp = igm_image_obj.pointcalc.lon_lat_alt_to_pixel_x_y(lon_lat_from_ortho[0], lon_lat_from_ortho[1], 0, crs_defs.PROJ_4326)



#

#
# geot_image.write_to_disk(os.path.join(output_dir, "tmp_geotiff.tif"))

#
# ul_ortho_kp_match = (0, 0)
# br_ortho_kp_match = (envi_ortho_header['lines']-1, envi_ortho_header['samples']-1)
#
