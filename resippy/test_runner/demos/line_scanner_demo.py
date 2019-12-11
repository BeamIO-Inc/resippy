from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.line_scanner import LineScannerPointCalc
from resippy.image_objects.image_factory import ImageFactory
from resippy.utils import proj_utils
from resippy.utils import photogrammetry_utils
import pyproj
from resippy.photogrammetry import crs_defs
from resippy.utils.image_utils import image_utils
import os

import numpy as np

import matplotlib.pyplot as plt


def simple_line_scanner_demo():
    nsamples = 200
    nlines = 500
    image_data = np.linspace(0, 255, nsamples*nlines)
    image_data = image_utils.unflatten_image_band(image_data, nsamples, nlines)
    line_scanner_point_calc = LineScannerPointCalc()

    cross_track_angles_degrees = np.linspace(-5, 5, nsamples)
    along_track_angle = 0

    lat = 38.8898
    lon = -77.0092
    alt = 100

    roll_begin_degrees = -2
    roll_end_degrees = 2
    roll_x_axis = np.linspace(np.deg2rad(roll_begin_degrees), np.deg2rad(roll_end_degrees), nlines)
    line_rolls = np.sin(roll_x_axis)

    line_m_matrices = []
    for line_roll in line_rolls:
        line_m_matrices.append(photogrammetry_utils.create_M_matrix(line_roll, 0, 0))

    line_scanner_point_calc.set_line_m_matrices(line_m_matrices)

    boresight_roll = np.deg2rad(0)
    boresight_pitch = np.deg2rad(0)
    boresight_yaw = np.deg2rad(0)

    boresight_matrix = photogrammetry_utils.create_M_matrix(boresight_roll, boresight_pitch, boresight_yaw)

    along_track_meters_per_line = 1

    local_proj = proj_utils.decimal_degrees_to_local_utm_proj(lon, lat)
    line_scanner_point_calc.set_projection(local_proj)
    local_start_lon, local_start_lat = pyproj.transform(crs_defs.PROJ_4326, local_proj, lon, lat)

    local_lons = np.zeros(nlines) + local_start_lon
    local_lats = np.zeros(nlines) + local_start_lat + np.linspace(0, along_track_meters_per_line*nlines, nlines)
    local_alts = np.zeros(nlines) + alt

    line_scanner_point_calc.set_line_utm_eastings(local_lons)
    line_scanner_point_calc.set_line_utm_northings(local_lats)
    line_scanner_point_calc.set_line_alts(local_alts)

    line_scanner_point_calc.set_pixel_cross_track_angles(cross_track_angles_degrees, angle_units='degrees')
    line_scanner_point_calc.set_pixel_along_track_angles(along_track_angle, angle_units='degrees')

    line_scanner_point_calc.set_boresight_matrix(boresight_matrix)

    line_scanner_image = ImageFactory.line_scanner.from_numpy_array_and_point_calc(image_data, line_scanner_point_calc)

    ortho_gtiff = line_scanner_image.ortho_entire_image(image_data, 500, 200)

    ortho_gtiff.write_to_disk(os.path.expanduser("~/Downloads/test_gtiff.tif"))

    stop = 1

def main():
    simple_line_scanner_demo()


if __name__ == '__main__':
    main()