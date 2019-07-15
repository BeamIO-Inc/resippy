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

    def get_nav_record(self, gps_time):
        if self._gps_time_in_range(gps_time):
            right_index = bisect_right(self._nav_data['gps_time'], gps_time)
            left_index = right_index - 1

            xs = [self._nav_data['gps_time'][left_index], self._nav_data['gps_time'][right_index]]

            return {key: np.interp(gps_time, xs, [value[left_index], value[right_index]])
                    for key, value in self._nav_data.items()}

        # TODO: throw exception/error instead of returning None
        return None
