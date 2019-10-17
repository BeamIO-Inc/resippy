import numpy as np
import json
import pdal
import pyproj

from resippy.photogrammetry.nav.abstract_nav import AbstractNav
from resippy.utils.string_utils import convert_to_snake_case


class ApplanixSBETNav(AbstractNav):

    def __init__(self
                 ):     # type: (...) -> ApplanixSBETNav
        super(ApplanixSBETNav, self).__init__()

    def load_from_file(self,
                       filename,    # type: str
                       proj,        # type: str
                       zone,        # type: str
                       ellps,       # type: str
                       datum        # type: str
                       ):           # type: (...) -> None
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
            self._projection = pyproj.Proj(proj=proj, ellps=ellps, datum=datum, preserve_units=True)
        else:
            self._projection = pyproj.Proj(proj=proj, zone=zone, ellps=ellps, datum=datum, preserve_units=True)

    def _gps_times_in_range(self,
                            gps_times   # type: np.ndarray
                            ):          # type: (...) -> bool
        if (self._nav_data['gps_time'][0] <= gps_times).all() and (self._nav_data['gps_time'][-1] >= gps_times).all():
            return True

        return False

    @staticmethod
    def _linear_interp(x,   # type: np.ndarray
                       xs,  # type: np.ndarray
                       ys   # type: np.ndarray
                       ):   # type: (...) -> np.ndarray
        return (ys[0, :]*(xs[1, :] - x) + ys[1, :]*(x - xs[0, :])) / (xs[1, :] - xs[0, :])

    def _get_indexes(self,
                     gps_times  # type: np.ndarray
                     ):         # type: (...) -> (np.ndarray, np.ndarray)
        right_indexes = np.searchsorted(self._nav_data['gps_time'], gps_times)
        left_indexes = right_indexes - 1

        return left_indexes, right_indexes

    def _get_xs(self,
                left_indexes,   # type: np.ndarray
                right_indexes   # type: np.ndarray
                ):              # type: (...) -> np.ndarray
        return np.array([self._nav_data['gps_time'][left_indexes], self._nav_data['gps_time'][right_indexes]])

    def _get_nav_records_native(self,
                                gps_times   # type: np.ndarray
                                ):  # type: (...) -> np.ndarray
        if self._gps_times_in_range(gps_times):
            left_indexes, right_indexes = self._get_indexes(gps_times)
            xs = self._get_xs(left_indexes, right_indexes)

            records = np.array([{key: value[i] for key, value in {
                key: self._linear_interp(gps_times, xs, np.array([value[left_indexes], value[right_indexes]])) for
                key, value in self._nav_data.items()}.items()} for i in range(gps_times.size)])

            return records

        # TODO: throw exception/error instead of returning None
        return None

    def _get_world_ys_native(self,
                             gps_times  # type: np.ndarray
                             ):         # type: (...) -> np.ndarray
        if self._gps_times_in_range(gps_times):
            left_indexes, right_indexes = self._get_indexes(gps_times)
            xs = self._get_xs(left_indexes, right_indexes)

            lats_left = self._nav_data['y'][left_indexes]
            lats_right = self._nav_data['y'][right_indexes]
            lats = self._linear_interp(gps_times, xs, np.array([lats_left, lats_right]))

            return lats

        # TODO: throw exception/error instead of returning None
        return None

    def _get_world_xs_native(self,
                             gps_times  # type: np.ndarray
                             ):         # type: (...) -> np.ndarray
        if self._gps_times_in_range(gps_times):
            left_indexes, right_indexes = self._get_indexes(gps_times)
            xs = self._get_xs(left_indexes, right_indexes)

            lons_left = self._nav_data['x'][left_indexes]
            lons_right = self._nav_data['x'][right_indexes]
            lons = self._linear_interp(gps_times, xs, np.array([lons_left, lons_right]))

            return lons

        # TODO: throw exception/error instead of returning None
        return None

    def _get_alts_native(self,
                         gps_times  # type: np.ndarray
                         ):         # type: (...) -> np.ndarray
        if self._gps_times_in_range(gps_times):
            left_indexes, right_indexes = self._get_indexes(gps_times)
            xs = self._get_xs(left_indexes, right_indexes)

            alts_left = self._nav_data['z'][left_indexes]
            alts_right = self._nav_data['z'][right_indexes]
            alts = self._linear_interp(gps_times, xs, np.array([alts_left, alts_right]))

            return alts

        # TODO: throw exception/error instead of returning None
        return None

    def _get_rolls_native(self,
                          gps_times     # type: np.ndarray
                          ):            # type: (...) -> np.ndarray
        if self._gps_times_in_range(gps_times):
            left_indexes, right_indexes = self._get_indexes(gps_times)
            xs = self._get_xs(left_indexes, right_indexes)

            rolls_left = self._nav_data['roll'][left_indexes]
            rolls_right = self._nav_data['roll'][right_indexes]
            rolls = self._linear_interp(gps_times, xs, np.array([rolls_left, rolls_right]))

            return rolls

        # TODO: throw exception/error instead of returning None
        return None

    def _get_pitches_native(self,
                            gps_times   # type: np.ndarray
                            ):          # type: (...) -> np.ndarray
        if self._gps_times_in_range(gps_times):
            left_indexes, right_indexes = self._get_indexes(gps_times)
            xs = self._get_xs(left_indexes, right_indexes)

            pitches_left = self._nav_data['pitch'][left_indexes]
            pitches_right = self._nav_data['pitch'][right_indexes]
            pitches = self._linear_interp(gps_times, xs, np.array([pitches_left, pitches_right]))

            return pitches

        # TODO: throw exception/error instead of returning None
        return None

    def _get_headings_native(self,
                             gps_times  # type: np.ndarray
                             ):         # type: (...) -> np.ndarray
        if self._gps_times_in_range(gps_times):
            left_indexes, right_indexes = self._get_indexes(gps_times)
            xs = self._get_xs(left_indexes, right_indexes)

            azimuths_left = self._nav_data['azimuth'][left_indexes]
            azimuths_right = self._nav_data['azimuth'][right_indexes]
            azimuths = self._linear_interp(gps_times, xs, np.array([azimuths_left, azimuths_right]))

            return azimuths

        # TODO: throw exception/error instead of returning None
        return None
