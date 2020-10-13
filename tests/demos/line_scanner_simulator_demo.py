import os

import numpy
from tests.demo_tests import demo_data_base_dir
from resippy.image_objects.image_factory import GeotiffImageFactory
from resippy.utils.linescanner_simulator import LinescannerSimulator
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.supporting_classes.linescanner_point_calc_factory import \
    LinescannerPointCalcFactory
from resippy.photogrammetry.lens_distortion_models.polynomial_model import PolynomialDistortionModel
from resippy.photogrammetry import ortho_tools
from resippy.photogrammetry.dem.dem_factory import DemFactory
from pyproj import transform
from resippy.photogrammetry import crs_defs

gtiff_fname = os.path.join(demo_data_base_dir,
                           "image_data/digital_globe/WashingtonDC_View-Ready_Stereo_50cm/056082264010/056082264010_01_P001_PAN/12SEP21160917-P2AS_R3C2-056082264010_01_P001.TIF")
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
focal_length_units = 'mm'
pixel_pitch = 5
pixel_pixel_units = 'micrometers'

lon_start = \
gtiff_basemap_image_obj.pointcalc.pixel_x_y_alt_to_lon_lat(gtiff_basemap_image_obj.metadata.get_npix_x() * 0.25, 0, 0)[
    0]
lon_end = \
gtiff_basemap_image_obj.pointcalc.pixel_x_y_alt_to_lon_lat(gtiff_basemap_image_obj.metadata.get_npix_x() * 0.75, 0, 0)[
    0]

lat_start = \
gtiff_basemap_image_obj.pointcalc.pixel_x_y_alt_to_lon_lat(0, gtiff_basemap_image_obj.metadata.get_npix_y() * 0.45, 0)[
    1]
lat_end = \
gtiff_basemap_image_obj.pointcalc.pixel_x_y_alt_to_lon_lat(0, gtiff_basemap_image_obj.metadata.get_npix_y() * 0.55, 0)[
    1]

lons = numpy.linspace(lon_start, lon_end, n_lines)
lats = numpy.linspace(lat_start, lat_start, n_lines)
alts = numpy.ones(n_lines) * sensor_alt

lons_dd, lats_dd = transform(gtiff_basemap_image_obj.pointcalc.get_projection(), crs_defs.PROJ_4326, lons, lats)

roll_setup = numpy.linspace(0, 2 * numpy.pi, len(lons))
rolls = numpy.sin(roll_setup) * 0.25
pitches = numpy.zeros(n_lines)
yaws = numpy.zeros(n_lines)
rpy_units = 'degrees'

coeffs = numpy.asarray([-0.00000460597723229289,
                        0.00065076473356702,
                        -0.00251815599654785,
                        0.00157290743703965,
                        -0.000473527702103131,
                        0.0000387085590987821
                        ])

poly_model = PolynomialDistortionModel(coeffs, flen, focal_length_units=focal_length_units)
point_calc = LinescannerPointCalcFactory.from_distortion_model_and_fpa_loc(poly_model,
                                                                           flen,
                                                                           focal_length_units,
                                                                           camera_npix_y,
                                                                           pixel_pitch,
                                                                           pixel_pitch,
                                                                           fpa_x_center=1,
                                                                           fpa_xy_center_units='mm')

point_calc.set_xyz_with_wgs84_coords(lons_dd, lats_dd, alts, 'meters')
point_calc.set_roll_pitch_yaws(rolls, pitches, yaws, units=rpy_units)

simulator = LinescannerSimulator(gtiff_fname, point_calc)
simulated_image_obj = simulator.create_overhead_image_object()

igm_image = ortho_tools.create_igm_image(simulated_image_obj, dem, dem_sample_distance=10)

image_lons = igm_image.pointcalc.lon_image
image_lats = igm_image.pointcalc.lat_image
image_alts = igm_image.pointcalc.alt_image

gtiff_image = ortho_tools.create_full_ortho_gtiff_image(igm_image)
gtiff_image.write_to_disk(os.path.expanduser("~/Downloads/gtiff_from_linescan_igm.tif"))

print("Check your downloads folder for the simulated camera results.")
