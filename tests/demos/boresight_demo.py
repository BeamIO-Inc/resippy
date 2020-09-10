import os

import numpy
from tests.demo_tests import demo_data_base_dir, demo_data_save_dir
from resippy.image_objects.image_factory import GeotiffImageFactory
from resippy.utils.physical_camera_simulator import PhysicalCameraSimulator
from resippy.utils.boresighter import SiftBoresighter
import pickle
from resippy.utils import file_utils
import matplotlib.pyplot as plt


gtiff_image = os.path.join(demo_data_base_dir, "image_data/digital_globe/WashingtonDC_View-Ready_Stereo_50cm/056082264010/056082264010_01_P001_PAN/12SEP21160917-P2AS_R3C2-056082264010_01_P001.TIF")
save_dir = os.path.join(demo_data_save_dir, "boresight_demo")
file_utils.make_dir_if_not_exists(save_dir)
sensor_alt_1 = 20000
sensor_alt_2 = 25000
camera_npix_x = 640
camera_npix_y = 480
flen = 60
pp = 5

gtiff_basemap_image_obj = GeotiffImageFactory.from_file(gtiff_image)

lon_center, lat_center = gtiff_basemap_image_obj.get_point_calculator().pixel_x_y_alt_to_lon_lat(
    gtiff_basemap_image_obj.get_metadata().get_npix_x() / 2,
    gtiff_basemap_image_obj.get_metadata().get_npix_y() / 2,
    0)

boresight_roll_offset_deg = 0.05
boresight_pitch_offset_deg = -0.03
boresight_yaw_offset_deg = 0

pass1_heading_from_east = 20

antiparallel_passes_xyzrpy = [[lon_center, lat_center, sensor_alt_1, 0, 0, pass1_heading_from_east],
                              [lon_center, lat_center, sensor_alt_1, 0.00, 0.00, pass1_heading_from_east + 180]]

antiparallel_passes_xyzrpy = numpy.array(antiparallel_passes_xyzrpy).transpose()

flightline_passes_xyzrpy = [[lon_center, lat_center, sensor_alt_1, 0, 0, pass1_heading_from_east],
                            [lon_center + 20, lat_center + 20, sensor_alt_1, 0.00, 0.00, pass1_heading_from_east + 90]]

flightline_passes_xyzrpy = numpy.array(flightline_passes_xyzrpy).transpose()


def create_simulated_image_objects(lons, lats, alts, rolls, pitches, yaws):
    simulator_zero_boresight = PhysicalCameraSimulator(gtiff_image, flen, camera_npix_x, camera_npix_y,)
    simulator_w_boresight = PhysicalCameraSimulator(gtiff_image,
                                                    flen,
                                                    camera_npix_x,
                                                    camera_npix_y,
                                                    boresight_roll=boresight_roll_offset_deg,
                                                    boresight_pitch=boresight_pitch_offset_deg,
                                                    boresight_yaw=boresight_yaw_offset_deg)

    simulated_image_objects = []
    for i in range(len(lons)):
        flightline_image_obj = simulator_w_boresight.create_overhead_image_object(lons[i],
                                                                                  lats[i],
                                                                                  alts[i],
                                                                                  rolls[i],
                                                                                  pitches[i],
                                                                                  yaws[i])

        flightline_zero_boresight = simulator_zero_boresight.create_overhead_image_object(lons[i],
                                                                                  lats[i],
                                                                                  alts[i],
                                                                                  rolls[i],
                                                                                  pitches[i],
                                                                                  yaws[i])

        flightline_image_obj.pointcalc = flightline_zero_boresight.pointcalc

        simulated_image_objects.append(flightline_image_obj)
    return simulated_image_objects


antiparallel_flight_image_objects = create_simulated_image_objects(antiparallel_passes_xyzrpy[0],
                                                                   antiparallel_passes_xyzrpy[1],
                                                                   antiparallel_passes_xyzrpy[2],
                                                                   antiparallel_passes_xyzrpy[3],
                                                                   antiparallel_passes_xyzrpy[4],
                                                                   antiparallel_passes_xyzrpy[5])

perpendicular_flight_image_objects = create_simulated_image_objects(flightline_passes_xyzrpy[0],
                                                                    flightline_passes_xyzrpy[1],
                                                                    flightline_passes_xyzrpy[2],
                                                                    flightline_passes_xyzrpy[3],
                                                                    flightline_passes_xyzrpy[4],
                                                                    flightline_passes_xyzrpy[5])

boresighter = SiftBoresighter()

rolls, pitches = boresighter.compute_search_space_rolls_and_pitches(antiparallel_flight_image_objects[0],
                                                                    antiparallel_flight_image_objects[1])

antiparallel_detection_planes = boresighter.create_boresight_detection_planes(antiparallel_flight_image_objects[0],
                                                                              antiparallel_flight_image_objects[1],
                                                                              rolls, pitches,
                                                                              numpy.array([-1, -1.3, -1.6, 0, 0.3, 0.6, 1]))

perpendicular_detection_planes = boresighter.create_boresight_detection_planes(perpendicular_flight_image_objects[0],
                                                                               perpendicular_flight_image_objects[1],
                                                                               rolls, pitches,
                                                                               numpy.array([-1, -1.3, -1.6, 0, 0.3, 0.6, 1]))

stop = 1