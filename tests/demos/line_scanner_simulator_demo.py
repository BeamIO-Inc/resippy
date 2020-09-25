import os

from tests.demo_tests import demo_data_base_dir
from resippy.image_objects.image_factory import GeotiffImageFactory
from resippy.utils.linescanner_simulator import LinescannerSimulator
from resippy.photogrammetry import ortho_tools
from resippy.photogrammetry.dem.dem_factory import DemFactory


gtiff_fname = os.path.join(demo_data_base_dir, "image_data/digital_globe/WashingtonDC_View-Ready_Stereo_50cm/056082264010/056082264010_01_P001_PAN/12SEP21160917-P2AS_R3C2-056082264010_01_P001.TIF")
gtiff_basemap_image_obj = GeotiffImageFactory.from_file(gtiff_fname)

dem = DemFactory.constant_elevation()

lon_center, lat_center = gtiff_basemap_image_obj.get_point_calculator().pixel_x_y_alt_to_lon_lat(
    gtiff_basemap_image_obj.get_metadata().get_npix_x() / 2,
    gtiff_basemap_image_obj.get_metadata().get_npix_y() / 2,
    0)

sensor_alt = 1000

camera_npix_x = 640
camera_npix_y = 480
flen = 3

yaw = 0.00000001

simulator = PhysicalCameraSimulator(gtiff_fname, flen, camera_npix_x, camera_npix_y)
pass1_image_obj = simulator.create_overhead_image_object(lon_center, lat_center, sensor_alt, 0, 0, yaw)

igm_image = ortho_tools.create_igm_image(pass1_image_obj, dem, dem_sample_distance=10)

image_lons = igm_image.pointcalc.lon_image
image_lats = igm_image.pointcalc.lat_image
image_alts = igm_image.pointcalc.alt_image

gtiff_image = ortho_tools.create_full_ortho_gtiff_image(igm_image)
gtiff_image.write_to_disk(os.path.expanduser("~/Downloads/gtiff_from_igm.tif"))

print("Check your downloads folder for the simulated camera results.")
