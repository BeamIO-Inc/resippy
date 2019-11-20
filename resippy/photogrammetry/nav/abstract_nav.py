import abc
import numbers
import numpy as np
from six import add_metaclass
from pyproj import Proj


@add_metaclass(abc.ABCMeta)
class AbstractNav:

    def __init__(self):
        self._num_records = 0
        self._record_length = 0
        self._nav_data = None
        self._projection = None

    @abc.abstractmethod
    def _get_nav_records_native(self,
                                gps_times   # type: np.ndarray
                                ):          # type: (...) -> np.ndarray
        pass

    @abc.abstractmethod
    def _get_world_ys_native(self,
                             gps_times  # type: np.ndarray
                             ):         # type: (...) -> np.ndarray
        pass

    @abc.abstractmethod
    def _get_world_xs_native(self,
                             gps_times  # type: np.ndarray
                             ):         # type: (...) -> np.ndarray
        pass

    @abc.abstractmethod
    def _get_alts_native(self,
                         gps_times  # type: np.ndarray
                         ):         # type: (...) -> np.ndarray
        pass

    @abc.abstractmethod
    def _get_rolls_native(self,
                          gps_times     # type: np.ndarray
                          ):            # type: (...) -> np.ndarray
        pass

    @abc.abstractmethod
    def _get_pitches_native(self,
                            gps_times   # type: np.ndarray
                            ):          # type: (...) -> np.ndarray
        pass

    @abc.abstractmethod
    def _get_headings_native(self,
                             gps_times  # type: np.ndarray
                             ):         # type: (...) -> np.ndarray
        pass

    def get_num_records(self
                        ):  # type: (...) -> int
        return self._num_records

    def get_record_length(self
                          ):  # type: (...) -> int
        return self._record_length

    def get_projection(self
                       ):   # type: (...) -> Proj
        return self._projection

    def get_nav_records(self,
                        gps_times   # type: np.ndarray
                        ):          # type: (...) -> np.ndarray
        gps_times, descriptor = self._format_input_array(gps_times)

        records = self._get_nav_records_native(gps_times)

        return self._format_output_array(records, descriptor)

    def get_world_ys(self,
                     gps_times  # type: np.ndarray
                     ):         # type: (...) -> np.ndarray
        gps_times, descriptor = self._format_input_array(gps_times)

        world_ys = self._get_world_ys_native(gps_times)

        return self._format_output_array(world_ys, descriptor)

    def get_world_xs(self,
                     gps_times  # type: np.ndarray
                     ):         # type: (...) -> np.ndarray
        gps_times, descriptor = self._format_input_array(gps_times)

        world_xs = self._get_world_xs_native(gps_times)

        return self._format_output_array(world_xs, descriptor)

    def get_alts(self,
                 gps_times  # type: np.ndarray
                 ):         # type: (...) -> np.ndarray
        gps_times, descriptor = self._format_input_array(gps_times)

        alts = self._get_alts_native(gps_times)

        return self._format_output_array(alts, descriptor)

    def get_rolls(self,
                  gps_times     # type: np.ndarray
                  ):            # type: (...) -> np.ndarray
        gps_times, descriptor = self._format_input_array(gps_times)

        rolls = self._get_rolls_native(gps_times)

        return self._format_output_array(rolls, descriptor)

    def get_pitches(self,
                    gps_times   # type: np.ndarray
                    ):          # type: (...) -> np.ndarray
        gps_times, descriptor = self._format_input_array(gps_times)

        pitches = self._get_pitches_native(gps_times)

        return self._format_output_array(pitches, descriptor)

    def get_headings(self,
                     gps_times  # type: np.ndarray
                     ):         # type: (...) -> np.ndarray
        gps_times, descriptor = self._format_input_array(gps_times)

        headings = self._get_headings_native(gps_times)

        return self._format_output_array(headings, descriptor)

    @staticmethod
    def _format_input_array(input_array     # type: np.ndarray
                            ):              # type: (...) -> (np.ndarray, dict)
        descriptor = {'input_array_is_number': isinstance(input_array, numbers.Number), 'input_array_is_2d': False,
                      'ny': 0, 'nx': 0}

        if descriptor['input_array_is_number']:
            input_array = np.array([input_array])

        if input_array.ndim == 2:
            descriptor['input_array_is_2d'] = True
            descriptor['ny'], descriptor['nx'] = input_array.shape
            input_array = input_array.reshape(descriptor['ny'] * descriptor['nx'])

        return input_array, descriptor

    @staticmethod
    def _format_output_array(output_array,  # type: np.ndarray
                             descriptor     # type: dict
                             ):             # type: (...) -> np.ndarray
        if output_array is None:
            return None

        if descriptor['input_array_is_number']:
            return output_array[0]
        elif descriptor['input_array_is_2d']:
            return output_array.reshape(descriptor['ny'], descriptor['nx'])

        return output_array
