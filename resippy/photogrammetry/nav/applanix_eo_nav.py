import numpy as np
import json
import pdal
import pyproj

from resippy.photogrammetry.nav.abstract_nav import AbstractNav
from resippy.utils.string_utils import convert_to_snake_case


class ApplanixEONav(AbstractNav):

    def __init__(self
                 ):     # type: (...) -> ApplanixEONav
        super(ApplanixEONav, self).__init__()

    def load_from_file(self,
                       filename,    # type: str
                       proj,        # type: str
                       zone,        # type: str
                       ellps,       # type: str
                       datum        # type: str
                       ):           # type: (...) -> None
        gps_time, easting, northing, height, omega, phi, kappa, lat, lon = [], [], [], [], [], [], [], [], []

        with open(filename) as f:
            for line in f:
                line_parts = line.split()

                try:
                    gps_time.append(float(line_parts[1]))
                    easting.append(float(line_parts[2]))
                    northing.append(float(line_parts[3]))
                    height.append(float(line_parts[4]))
                    omega.append(float(line_parts[5]))
                    phi.append(float(line_parts[6]))
                    kappa.append(float(line_parts[7]))
                    lat.append(float(line_parts[8]))
                    lon.append(float(line_parts[9]))
                except (IndexError, ValueError):
                    pass

        self._nav_data = {
            'gps_time': np.array(gps_time),
            'easting': np.array(easting),
            'northing': np.array(northing),
            'height': np.array(height),
            'omega': np.array(omega),
            'phi': np.array(phi),
            'kappa': np.array(kappa),
            'lat': np.array(lat),
            'lon': np.array(lon)
        }

        self._num_records = len(gps_time)
        self._record_length = 7

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
                             gps_times  # type: np.ndarray,
                             ):         # type: (...) -> np.ndarray
        if self._gps_times_in_range(gps_times):
            left_indexes, right_indexes = self._get_indexes(gps_times)
            xs = self._get_xs(left_indexes, right_indexes)

            northings_left = self._nav_data['northing'][left_indexes]
            northings_right = self._nav_data['northing'][right_indexes]
            northings = self._linear_interp(gps_times, xs, np.array([northings_left, northings_right]))

            return northings

        # TODO: throw exception/error instead of returning None
        return None

    def _get_world_xs_native(self,
                             gps_times  # type: np.ndarray
                             ):         # type: (...) -> np.ndarray
        if self._gps_times_in_range(gps_times):
            left_indexes, right_indexes = self._get_indexes(gps_times)
            xs = self._get_xs(left_indexes, right_indexes)

            eastings_left = self._nav_data['easting'][left_indexes]
            eastings_right = self._nav_data['easting'][right_indexes]
            eastings = self._linear_interp(gps_times, xs, np.array([eastings_left, eastings_right]))

            return eastings

        # TODO: throw exception/error instead of returning None
        return None

    def _get_alts_native(self,
                         gps_times  # type: np.ndarray
                         ):         # type: (...) -> np.ndarray
        if self._gps_times_in_range(gps_times):
            left_indexes, right_indexes = self._get_indexes(gps_times)
            xs = self._get_xs(left_indexes, right_indexes)

            heights_left = self._nav_data['height'][left_indexes]
            heights_right = self._nav_data['height'][right_indexes]
            heights = self._linear_interp(gps_times, xs, np.array([heights_left, heights_right]))

            return heights

        # TODO: throw exception/error instead of returning None
        return None

    def _get_rolls_native(self,
                          gps_times     # type: np.ndarray
                          ):            # type: (...) -> np.ndarray
        if self._gps_times_in_range(gps_times):
            left_indexes, right_indexes = self._get_indexes(gps_times)
            xs = self._get_xs(left_indexes, right_indexes)

            omegas_left = self._nav_data['omega'][left_indexes]
            omegas_right = self._nav_data['omega'][right_indexes]
            omegas = self._linear_interp(gps_times, xs, np.array([omegas_left, omegas_right]))

            return omegas

        # TODO: throw exception/error instead of returning None
        return None

    def _get_pitches_native(self,
                            gps_times   # type: np.ndarray
                            ):          # type: (...) -> np.ndarray
        if self._gps_times_in_range(gps_times):
            left_indexes, right_indexes = self._get_indexes(gps_times)
            xs = self._get_xs(left_indexes, right_indexes)

            phis_left = self._nav_data['phi'][left_indexes]
            phis_right = self._nav_data['phi'][right_indexes]
            phis = self._linear_interp(gps_times, xs, np.array([phis_left, phis_right]))

            return phis

        # TODO: throw exception/error instead of returning None
        return None

    def _get_headings_native(self,
                             gps_times  # type: np.ndarray
                             ):         # type: (...) -> np.ndarray
        if self._gps_times_in_range(gps_times):
            left_indexes, right_indexes = self._get_indexes(gps_times)
            xs = self._get_xs(left_indexes, right_indexes)

            kappas_left = self._nav_data['kappa'][left_indexes]
            kappas_right = self._nav_data['kappa'][right_indexes]
            kappas = self._linear_interp(gps_times, xs, np.array([kappas_left, kappas_right]))

            return kappas

        # TODO: throw exception/error instead of returning None
        return None
