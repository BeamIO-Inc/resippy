import os

from tests.demo_tests import demo_data_base_dir
from resippy.image_objects.image_factory import GeotiffImageFactory
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.fixtured_camera import FixturedCamera
from resippy.photogrammetry import ortho_tools
from resippy.utils.physical_camera_simulator import PhysicalCameraSimulator
from resippy.utils.boresighter import SiftBoresighter


gtiff_image = os.path.join(demo_data_base_dir, "image_data/digital_globe/WashingtonDC_View-Ready_Stereo_50cm/056082264010/056082264010_01_P001_PAN/12SEP21160917-P2AS_R3C2-056082264010_01_P001.TIF")

gtiff_basemap_image_obj = GeotiffImageFactory.from_file(gtiff_image)

lon_center, lat_center = gtiff_basemap_image_obj.get_point_calculator().pixel_x_y_alt_to_lon_lat(
    gtiff_basemap_image_obj.get_metadata().get_npix_x() / 2,
    gtiff_basemap_image_obj.get_metadata().get_npix_y() / 2,
    0)

sensor_alt_1 = 20000
sensor_alt_2 = 25000

print(lon_center, lat_center)

fixtured_camera = FixturedCamera()
fixtured_camera_2 = FixturedCamera()

camera_npix_x = 640
camera_npix_y = 480
flen = 60
pp = 5

boresight_roll_offset_deg = 0.05
boresight_pitch_offset_deg = 0
boresight_yaw_offset_deg = 0

# flightline_passes_xyzrpy = [[lon_center, lat_center, sensor_alt_1, 0, 0, 0],
#                             [lon_center, lat_center, sensor_alt_1, 0, 0, 180],
#                             [lon_center, lat_center, sensor_alt_1, 0, 0, 90],
#                             [lon_center, lat_center, sensor_alt_1, 0, 0, 270],
#                             [lon_center, lat_center, sensor_alt_2, 0, 0, 0],
#                             [lon_center, lat_center, sensor_alt_2, 0, 0, 180],
#                             [lon_center, lat_center, sensor_alt_2, 0, 0, 90],
#                             [lon_center, lat_center, sensor_alt_2, 0, 0, 270]]

flightline_passes_xyzrpy = [[lon_center, lat_center, sensor_alt_1, 0, 0, 0],
                            [lon_center, lat_center, sensor_alt_1, 0, 0, 180]]

simulator_zero_boresight = PhysicalCameraSimulator(gtiff_image, flen, camera_npix_x, camera_npix_y,)
simulator_w_boresight = PhysicalCameraSimulator(gtiff_image,
                                                flen,
                                                camera_npix_x,
                                                camera_npix_y,
                                                boresight_roll=boresight_roll_offset_deg,
                                                boresight_pitch=boresight_pitch_offset_deg,
                                                boresight_yaw=boresight_yaw_offset_deg)

flightline_image_objects = []
for i, flightline_pass in enumerate(flightline_passes_xyzrpy):
    flightline_image_obj = simulator_w_boresight.create_overhead_image_object(flightline_pass[0],
                                                                              flightline_pass[1],
                                                                              flightline_pass[2],
                                                                              flightline_pass[3],
                                                                              flightline_pass[4],
                                                                              flightline_pass[5])

    flightline_zero_boresight = simulator_zero_boresight.create_overhead_image_object(flightline_pass[0],
                                                                                      flightline_pass[1],
                                                                                      flightline_pass[2],
                                                                                      flightline_pass[3],
                                                                                      flightline_pass[4],
                                                                                      flightline_pass[5])

    flightline_image_obj.pointcalc = flightline_zero_boresight.pointcalc
    flightline_image_objects.append(flightline_image_obj)

boresighter = SiftBoresighter(flightline_image_objects)
boresighter.compoute_boresights()
