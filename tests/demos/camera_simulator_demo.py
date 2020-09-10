import os

from tests.demo_tests import demo_data_base_dir
from resippy.image_objects.image_factory import GeotiffImageFactory
from resippy.utils.physical_camera_simulator import PhysicalCameraSimulator
from resippy.utils.image_utils import image_utils
import imageio

import numpy

gtiff_fname = os.path.join(demo_data_base_dir, "image_data/digital_globe/WashingtonDC_View-Ready_Stereo_50cm/056082264010/056082264010_01_P001_PAN/12SEP21160917-P2AS_R3C2-056082264010_01_P001.TIF")

gtiff_basemap_image_obj = GeotiffImageFactory.from_file(gtiff_fname)

lon_center, lat_center = gtiff_basemap_image_obj.get_point_calculator().pixel_x_y_alt_to_lon_lat(
    gtiff_basemap_image_obj.get_metadata().get_npix_x() / 2,
    gtiff_basemap_image_obj.get_metadata().get_npix_y() / 2,
    0)

sensor_alt = 20000

camera_npix_x = 640
camera_npix_y = 480
flen = 60

yaws = numpy.arange(0, 361, 45/2) + 0.00001

simulator = PhysicalCameraSimulator(gtiff_fname, flen, camera_npix_x, camera_npix_y)
for i, yaw in enumerate(yaws):
    print("percent complete: " + str(i/len(yaws) * 100))
    pass1_image_obj = simulator.create_overhead_image_object(lon_center, lat_center, sensor_alt, 0, 0, yaw)
    cmapped_image = image_utils.apply_colormap_to_grayscale_image(pass1_image_obj.get_image_band(0))
    imageio.imsave(os.path.expanduser("~/Downloads/simulated_" + str.zfill(str(i+1), 5)) + ".tif", cmapped_image)

print("Check your downloads folder for the simulated camera results.")
