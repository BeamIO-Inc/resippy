from bisect import bisect_right
import numpy as np
import json
import pdal

from resippy.photogrammetry.nav.abstract_nav import AbstractNav
from resippy.utils.string_utils import convert_to_snake_case


class ApplanixSBETNav(AbstractNav):

    def __init__(self):
        super(ApplanixSBETNav, self).__init__()

    def load_from_file(self, filename):
        pipe_json = json.dumps([
            {
                'type': 'readers.sbet',
                'filename': filename
            }
        ])

        sbet_pipeline = pdal.Pipeline(pipe_json)
        sbet_pipeline.validate()
        self._num_records = sbet_pipeline.execute()

        data = sbet_pipeline.arrays[0]
        self._record_length = len(data.dtype.names)

        # convert structured data array to dict
        self._nav_data = {convert_to_snake_case(name): data[name] for name in data.dtype.names}

    def _gps_times_in_range(self, gps_times):
        if (self._nav_data['gps_time'][0] <= gps_times).all() and (self._nav_data['gps_time'][-1] >= gps_times).all():
            return True

        return False

    def _get_nav_records_native(self, gps_times):
        if self._gps_times_in_range(gps_times):
            right_indexes = bisect_right(self._nav_data['gps_time'], gps_times)
            left_indexes = right_indexes - 1

            xs = [self._nav_data['gps_time'][left_indexes], self._nav_data['gps_time'][right_indexes]]

            return {key: np.interp(gps_times, xs, [value[left_indexes], value[right_indexes]])
                    for key, value in self._nav_data.items()}

        # TODO: throw exception/error instead of returning None
        return None

    def _get_lats_native(self, gps_times):
        pass

    def _get_lons_native(self, gps_times):
        pass

    def _get_alts_native(self, gps_times):
        pass

    def _get_rolls_native(self, gps_times):
        pass

    def _get_pitches_native(self, gps_times):
        pass

    def _get_headings_native(self, gps_times):
        pass
