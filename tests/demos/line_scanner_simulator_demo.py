import os

import numpy
from tests.demo_tests import demo_data_base_dir
from resippy.image_objects.image_factory import GeotiffImageFactory
from resippy.utils.linescanner_simulator import LinescannerSimulator
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.line_scanner_point_calc import LineScannerPointCalc
from resippy.utils.image_utils import image_utils
from resippy.photogrammetry import ortho_tools
from resippy.photogrammetry.dem.dem_factory import DemFactory
from pyproj import transform
from resippy.photogrammetry import crs_defs


gtiff_fname = os.path.join(demo_data_base_dir, "image_data/digital_globe/WashingtonDC_View-Ready_Stereo_50cm/056082264010/056082264010_01_P001_PAN/12SEP21160917-P2AS_R3C2-056082264010_01_P001.TIF")
gtiff_basemap_image_obj = GeotiffImageFactory.from_file(gtiff_fname)

dem = DemFactory.constant_elevation()

lon_center, lat_center = gtiff_basemap_image_obj.get_point_calculator().pixel_x_y_alt_to_lon_lat(
    gtiff_basemap_image_obj.get_metadata().get_npix_x() / 2,
    gtiff_basemap_image_obj.get_metadata().get_npix_y() / 2,
    0)

n_lines = 600
camera_npix_y = 480
sensor_alt = 20000

flen = 60
pp = 5e-6

lon_start = gtiff_basemap_image_obj.pointcalc.pixel_x_y_alt_to_lon_lat(gtiff_basemap_image_obj.metadata.get_npix_x()*0.25, 0, 0)[0]
lon_end = gtiff_basemap_image_obj.pointcalc.pixel_x_y_alt_to_lon_lat(gtiff_basemap_image_obj.metadata.get_npix_x()*0.75, 0, 0)[0]

lat_start = gtiff_basemap_image_obj.pointcalc.pixel_x_y_alt_to_lon_lat(0, gtiff_basemap_image_obj.metadata.get_npix_y()*0.45, 0)[1]
lat_end = gtiff_basemap_image_obj.pointcalc.pixel_x_y_alt_to_lon_lat(0, gtiff_basemap_image_obj.metadata.get_npix_y()*0.55, 0)[1]

lons = numpy.linspace(lon_start, lon_end, n_lines)
lats = numpy.linspace(lat_start, lat_end, n_lines)
alts = numpy.ones(n_lines) * sensor_alt

lons_dd, lats_dd = transform(gtiff_basemap_image_obj.pointcalc.get_projection(), crs_defs.PROJ_4326, lons, lats)

rolls = numpy.zeros(n_lines)
pitches = numpy.zeros(n_lines)
yaws = numpy.zeros(n_lines)

point_calc = LineScannerPointCalc()
point_calc.set_camera_model_w_ideal_pinhole_model(camera_npix_y, flen, pp, focal_length_units='mm', pixel_pitch_units='meters')
point_calc.set_xyz_with_wgs84_coords(lons_dd, lats_dd, alts, 'meters')
point_calc.set_roll_pitch_yaws(rolls, pitches, yaws, units='degrees')

simulator = LinescannerSimulator(gtiff_fname, point_calc)
simulated_image_obj = simulator.create_overhead_image_object()

igm_image = ortho_tools.create_igm_image(simulated_image_obj, dem, dem_sample_distance=10)

image_lons = igm_image.pointcalc.lon_image
image_lats = igm_image.pointcalc.lat_image
image_alts = igm_image.pointcalc.alt_image

gtiff_image = ortho_tools.create_full_ortho_gtiff_image(igm_image)
gtiff_image.write_to_disk(os.path.expanduser("~/Downloads/gtiff_from_linescan_igm.tif"))

print("Check your downloads folder for the simulated camera results.")
