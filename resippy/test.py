import os
import json
import pyproj

from resippy.photogrammetry.nav.nav_factory import NavFactory
from resippy.image_objects.image_factory import ImageFactory

# open nav
sbet_filename = '/home/ryan/Data/dirs/20180823_snapbeans/Missions/1018/gpsApplanix/processed/sbet_1018.out'
applanix_nav = NavFactory.from_applanix_sbet_file(sbet_filename, 'utm', '18N', 'WGS84', 'WGS84')

# open micasense
micasense_base_dir = '/home/ryan/Data/dirs/20180823_snapbeans/Missions/1018/micasense/processed/merged'
camera_param_file = '/home/ryan/Code/dirs/dirs_algorithms/algorithms/micasense_open_image/micasense_parameters.json'
image_number = 175
product_level = 'L0'

bundle_name = product_level + '_IMG_' + str(image_number).zfill(4)

band_fname_dict = {}
for cam_num in range(1, 6):
    band_num = str(cam_num)
    frame_name = bundle_name + '_' + band_num + '.tif'

    band_fname_dict['band' + band_num] = os.path.join(micasense_base_dir, frame_name)
    per = float(cam_num) / 6.0 * 100.0

with open(camera_param_file) as json_file:
    json_data = json.load(json_file)
opencv_params = json_data['parameters']

micasense_image = ImageFactory.micasense.from_image_number_and_opencv(band_fname_dict, opencv_params)
micasense_image.get_metadata().set_image_name(bundle_name)

for band_num in range(5):
    exif = micasense_image.get_gps_timestamp_and_center(band_num)
    record = applanix_nav.get_nav_records(exif['gps_timestamp'])
    center = exif['center']

    lla_projection = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84', preserve_units=True)
    utm_projection = applanix_nav.get_projection()

    x_utm, y_utm, z_utm = pyproj.transform(lla_projection, utm_projection, record['x'], record['y'], record['z'])
    x_center_utm, y_center_utm = pyproj.transform(lla_projection, utm_projection, center['lon'], center['lat'])

    point_calc = micasense_image.get_point_calculator().get_point_calc(band_num)
    point_calc.init_extrinsic(x_utm, y_utm, z_utm, record['roll'], record['pitch'], record['azimuth'])
    point_calc.set_approximate_lon_lat_center(x_center_utm, y_center_utm)

sensor_model = micasense_image.get_point_calculator()

for point_calc in sensor_model.get_point_calcs():
    center = point_calc.get_approximate_lon_lat_center()

# import resippy.utils.time_utils as time_utils
# from datetime import datetime, timezone
#
# year = 2019
# month = 7
# day = 8
# hour = 16
# minute = 26
# second = 4
#
# utc_timestamp = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc).timestamp()
# __, seconds_in_week = time_utils.utc_timestamp_to_gps_week_and_seconds(utc_timestamp)
#
# timestamp_decimal = 0.800311379
#
# print(seconds_in_week + timestamp_decimal)
