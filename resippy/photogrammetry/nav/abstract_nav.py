import abc
import numbers
import numpy as np
from six import add_metaclass


@add_metaclass(abc.ABCMeta)
class AbstractNav:

    def __init__(self):
        self._num_records = 0
        self._record_length = 0
        self._nav_data = None

    @abc.abstractmethod
    def _get_nav_records_native(self,
                                gps_times   # type: np.ndarray
                                ):          # type: (...) -> np.ndarray
        pass

    @abc.abstractmethod
    def _get_lats_native(self,
                         gps_times  # type: np.ndarray
                         ):         # type: (...) -> np.ndarray
        pass

    @abc.abstractmethod
    def _get_lons_native(self,
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

    def get_nav_records(self,
                        gps_times   # type: np.ndarray
                        ):          # type: (...) -> np.ndarray
        gps_times, descriptor = self._format_input_array(gps_times)

        records = self._get_nav_records_native(gps_times)

        return self._format_output_array(records, descriptor)

    def get_lats(self,
                 gps_times  # type: np.ndarray
                 ):         # type: (...) -> np.ndarray
        pass

    def get_lons(self,
                 gps_times  # type: np.ndarray
                 ):         # type: (...) -> np.ndarray
        pass

    def get_alts(self,
                 gps_times  # type: np.ndarray
                 ):         # type: (...) -> np.ndarray
        pass

    def get_rolls(self,
                  gps_times     # type: np.ndarray
                  ):            # type: (...) -> np.ndarray
        pass

    def get_pitches(self,
                    gps_times   # type: np.ndarray
                    ):          # type: (...) -> np.ndarray
        pass

    def get_headings(self,
                     gps_times  # type: np.ndarray
                     ):         # type: (...) -> np.ndarray
        pass

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
        if descriptor['input_array_is_number']:
            return output_array[0]
        elif descriptor['input_array_is_2d']:
            return output_array.reshape(descriptor['ny'], descriptor['nx'])
