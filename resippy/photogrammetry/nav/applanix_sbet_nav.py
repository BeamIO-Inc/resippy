import numpy as np
import json
import pdal
import pyproj

from resippy.photogrammetry.nav.abstract_nav import AbstractNav
from resippy.utils.string_utils import convert_to_snake_case


class ApplanixSBETNav(AbstractNav):

    def __init__(self):
        super(ApplanixSBETNav, self).__init__()

        self._lla_projection = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

    def load_from_file(self, filename, proj, zone, ellps, datum):
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

        if not zone:
            self._projection = pyproj.Proj(proj=proj, ellps=ellps, datum=datum)
        else:
            self._projection = pyproj.Proj(proj=proj, zone=zone, ellps=ellps, datum=datum)

    def _gps_times_in_range(self, gps_times):
        if (self._nav_data['gps_time'][0] <= gps_times).all() and (self._nav_data['gps_time'][-1] >= gps_times).all():
            return True

        return False

    @staticmethod
    def _linear_interp(x, xs, ys):
        return (ys[0, :]*(xs[1, :] - x) + ys[1, :]*(x - xs[0, :])) / (xs[1, :] - xs[0, :])

    def _get_indexes(self, gps_times):
        right_indexes = np.searchsorted(self._nav_data['gps_time'], gps_times)
        left_indexes = right_indexes - 1

        return left_indexes, right_indexes

    def _get_xs(self, left_indexes, right_indexes):
        return np.array([self._nav_data['gps_time'][left_indexes], self._nav_data['gps_time'][right_indexes]])

    def _get_nav_records_native(self, gps_times):
        if self._gps_times_in_range(gps_times):
            left_indexes, right_indexes = self._get_indexes(gps_times)
            xs = self._get_xs(left_indexes, right_indexes)

            records = np.array([{key: value[i] for key, value in {
                key: self._linear_interp(gps_times, xs, np.array([value[left_indexes], value[right_indexes]])) for
                key, value in self._nav_data.items()}.items()} for i in range(gps_times.size)])

            return records

        # TODO: throw exception/error instead of returning None
        return None

    def _get_lats_native(self, gps_times):
        if self._gps_times_in_range(gps_times):
            left_indexes, right_indexes = self._get_indexes(gps_times)
            xs = self._get_xs(left_indexes, right_indexes)

            x_utm_left = self._nav_data['x'][left_indexes]
            x_utm_right = self._nav_data['x'][right_indexes]
            x_utm = self._linear_interp(gps_times, xs, np.array([x_utm_left, x_utm_right]))

            y_utm_left = self._nav_data['y'][left_indexes]
            y_utm_right = self._nav_data['y'][right_indexes]
            y_utm = self._linear_interp(gps_times, xs, np.array([y_utm_left, y_utm_right]))

            __, lats = pyproj.transform(self._projection, self._lla_projection, x_utm, y_utm)

            return lats

        # TODO: throw exception/error instead of returning None
        return None

    def _get_lons_native(self, gps_times):
        if self._gps_times_in_range(gps_times):
            left_indexes, right_indexes = self._get_indexes(gps_times)
            xs = self._get_xs(left_indexes, right_indexes)

            x_utm_left = self._nav_data['x'][left_indexes]
            x_utm_right = self._nav_data['x'][right_indexes]
            x_utm = self._linear_interp(gps_times, xs, np.array([x_utm_left, x_utm_right]))

            y_utm_left = self._nav_data['y'][left_indexes]
            y_utm_right = self._nav_data['y'][right_indexes]
            y_utm = self._linear_interp(gps_times, xs, np.array([y_utm_left, y_utm_right]))

            lons, __ = pyproj.transform(self._projection, self._lla_projection, x_utm, y_utm)

            return lons

        # TODO: throw exception/error instead of returning None
        return None

    def _get_alts_native(self, gps_times):
        if self._gps_times_in_range(gps_times):
            left_indexes, right_indexes = self._get_indexes(gps_times)
            xs = self._get_xs(left_indexes, right_indexes)

            alts_left = self._nav_data['z'][left_indexes]
            alts_right = self._nav_data['z'][right_indexes]
            alts = self._linear_interp(gps_times, xs, np.array([alts_left, alts_right]))

            return alts

        # TODO: throw exception/error instead of returning None
        return None

    def _get_rolls_native(self, gps_times):
        if self._gps_times_in_range(gps_times):
            left_indexes, right_indexes = self._get_indexes(gps_times)
            xs = self._get_xs(left_indexes, right_indexes)

            rolls_left = self._nav_data['roll'][left_indexes]
            rolls_right = self._nav_data['roll'][right_indexes]
            rolls = self._linear_interp(gps_times, xs, np.array([rolls_left, rolls_right]))

            return rolls

        # TODO: throw exception/error instead of returning None
        return None

    def _get_pitches_native(self, gps_times):
        if self._gps_times_in_range(gps_times):
            left_indexes, right_indexes = self._get_indexes(gps_times)
            xs = self._get_xs(left_indexes, right_indexes)

            pitches_left = self._nav_data['pitch'][left_indexes]
            pitches_right = self._nav_data['pitch'][right_indexes]
            pitches = self._linear_interp(gps_times, xs, np.array([pitches_left, pitches_right]))

            return pitches

        # TODO: throw exception/error instead of returning None
        return None

    def _get_headings_native(self, gps_times):
        if self._gps_times_in_range(gps_times):
            left_indexes, right_indexes = self._get_indexes(gps_times)
            xs = self._get_xs(left_indexes, right_indexes)

            azimuths_left = self._nav_data['azimuth'][left_indexes]
            azimuths_right = self._nav_data['azimuth'][right_indexes]
            azimuths = self._linear_interp(gps_times, xs, np.array([azimuths_left, azimuths_right]))

            return azimuths

        # TODO: throw exception/error instead of returning None
        return None
