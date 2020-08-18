import time
from datetime import datetime, timezone

from resippy.image_objects.abstract_image_metadata import AbstractImageMetadata
import resippy.utils.envi_utils as envi_utils
import resippy.utils.time_utils as time_utils


class MakoMetadata(AbstractImageMetadata):

    def __init__(self):
        super(MakoMetadata, self).__init__()
        self.envi_header = {}

    @classmethod
    def init_from_header(cls,
                         header_fname
                         ):
        metadata = cls()

        header = envi_utils.read_envi_header(header_fname)

        metadata.set_npix_x(header['samples'])
        metadata.set_npix_y(header['lines'])
        metadata.set_n_bands(header['bands'])
        metadata.envi_header = header

        return metadata

    def get_gps_timestamp(self):
        time_str = self.envi_header['acquisition time']
        time_struct = time.strptime(time_str, '%Y-%m-%dT%H:%M:%S.%f')
        timestamp = datetime(*time_struct[:6], tzinfo=timezone.utc).timestamp()
        timestamp_decimal = float('0.{}'.format(int(time_str.split('.')[-1])))
        utc_timestamp = timestamp + timestamp_decimal
        __, gps_time = time_utils.utc_timestamp_with_leap_to_gps_week_and_seconds(utc_timestamp)
        return gps_time

    def get_lon_lat_center(self):
        lat, lon = self.envi_header['geo points'].split(',')[2:]
        return [float(lon), float(lat)]
