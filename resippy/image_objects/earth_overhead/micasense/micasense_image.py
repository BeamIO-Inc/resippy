from __future__ import division

from numpy import ndarray
import numpy as np
from imageio import imread
import exifread
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

    def get_gps_timestamp_and_center(self,
                                     band_number    # type: int
                                     ):             # type: (...) -> dict
        with open(self.band_fnames[band_number], 'rb') as f:
            exif_data = exifread.process_file(f, details=False)

        utc_datetime_str = exif_data['EXIF DateTimeOriginal'].values
        year = int(utc_datetime_str[0:4])
        month = int(utc_datetime_str[5:7])
        day = int(utc_datetime_str[8:10])
        hour = int(utc_datetime_str[11:13])
        minute = int(utc_datetime_str[14:16])
        second = int(utc_datetime_str[17:19])

        utc_timestamp = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc).timestamp()
        __, seconds_in_week = time_utils.utc_timestamp_to_gps_week_and_seconds(utc_timestamp)

        timestamp_decimal_str = str(exif_data['EXIF SubSecTime'])
        timestamp_decimal = float('0.{}'.format(int(timestamp_decimal_str)))

        output_dict = {'gps_timestamp': seconds_in_week + timestamp_decimal}

        lon_tag = exif_data['GPS GPSLongitude']
        lon_ref_tag = exif_data['GPS GPSLongitudeRef']
        lon_dd = MicasenseImage._dms_tag_to_dd(lon_tag, lon_ref_tag)

        lat_tag = exif_data['GPS GPSLatitude']
        lat_ref_tag = exif_data['GPS GPSLatitudeRef']
        lat_dd = MicasenseImage._dms_tag_to_dd(lat_tag, lat_ref_tag)

        output_dict['center'] = {'lon': lon_dd, 'lat': lat_dd}

        return output_dict

    @staticmethod
    def _dms_tag_to_dd(dms_tag,     # type: exifread.classes.IfdTag
                       ref_tag      # type: exifread.classes.IfdTag
                       ):           # type: (...) -> float
        dms = dms_tag.values
        ref = ref_tag.values

        d = dms[0].num
        m = dms[1].num
        s = dms[2].num / dms[2].den

        sign = -1 if ref == 'S' or ref == 'W' else 1

        return sign * (int(d) + float(m) / 60 + float(s) / 3600)
