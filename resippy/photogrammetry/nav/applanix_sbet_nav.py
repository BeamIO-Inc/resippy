import numpy as np
import json
import pdal

from pyproj import Proj

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
        
        self._projection = Proj(proj='geocent', ellps='WGS84', datum='WGS84')

    def _gps_times_in_range(self, gps_times):
        if (self._nav_data['gps_time'][0] <= gps_times).all() and (self._nav_data['gps_time'][-1] >= gps_times).all():
            return True

        return False

    @staticmethod
    def _linear_interp(x, xs, ys):
        return (ys[0, :]*(xs[1, :] - x) + ys[1, :]*(x - xs[0, :])) / (xs[1, :] - xs[0, :])

    def _get_nav_records_native(self, gps_times):
        if self._gps_times_in_range(gps_times):
            right_indexes = np.searchsorted(self._nav_data['gps_time'], gps_times)
            left_indexes = right_indexes - 1

            xs = np.array([self._nav_data['gps_time'][left_indexes], self._nav_data['gps_time'][right_indexes]])

            records = np.array([{key: value[i] for key, value in {
                key: self._linear_interp(gps_times, xs, np.array([value[left_indexes], value[right_indexes]])) for
                key, value in self._nav_data.items()}.items()} for i in range(gps_times.size)])

            return records

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


if __name__ == '__main__':
    nav = ApplanixSBETNav()
    nav.load_from_file('/home/ryan/Data/dirs/20180823_snapbeans/Missions/1018/gpsApplanix/processed/sample.sbet')

    # times = 335010
    # times = np.array([335010, 335090, 335091])
    times = np.array([[335010, 335090], [335011, 335091.2]])

    records = nav.get_nav_records(times)
    print(f'nav: {records}')
    print(f'time-shape: {times.shape}')
    print(f'nav-shape: {records.shape}')
