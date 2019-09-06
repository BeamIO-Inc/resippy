from __future__ import division

from numpy import ndarray
import numpy as np
from imageio import imread
# import exiftool
from datetime import datetime, timezone

from resippy.image_objects.earth_overhead.abstract_earth_overhead_image import AbstractEarthOverheadImage
import resippy.utils.time_utils as time_utils


class MicasenseImage(AbstractEarthOverheadImage):
    def __init__(self):
        super(MicasenseImage, self).__init__()
        self.band_fnames = []

    def read_all_image_data_from_disk(self):  # type: (...) -> ndarray
        imgs = [imread(fname) for fname in self.band_fnames]
        all_image_data = np.dstack(imgs)

        return all_image_data

    def read_band_from_disk(self,
                            band_number     # type: int
                            ):              # type: (...) -> ndarray
        return imread(self.band_fnames[band_number])

    def get_gps_seconds_in_week(self,
                                band_number     # type: int
                                ):              # type: (...) -> float
        with exiftool.ExifTool() as exif_tool:
            exif_data = exif_tool.get_metadata(self.band_fnames[band_number])

        utc_datetime_str = exif_data['EXIF:DateTimeOriginal']
        year = int(utc_datetime_str[0:4])
        month = int(utc_datetime_str[5:7])
        day = int(utc_datetime_str[8:10])
        hour = int(utc_datetime_str[11:13])
        minute = int(utc_datetime_str[14:16])
        second = int(utc_datetime_str[17:19])

        utc_timestamp = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc).timestamp()
        __, seconds_in_week = time_utils.utc_timestamp_to_gps_week_and_seconds(utc_timestamp)

        timestamp_decimal_str = exif_data['EXIF:SubSecTime']
        timestamp_decimal = float('0.{}'.format(int(timestamp_decimal_str)))

        return seconds_in_week + timestamp_decimal
