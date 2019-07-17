import abc
import numpy as np
from six import add_metaclass


@add_metaclass(abc.ABCMeta)
class AbstractNav:

    def __init__(self):
        self._num_records = 0
        self._record_length = 0
        self._nav_data = None

    def get_num_records(self
                        ):  # type: (...) -> int
        return self._num_records

    def get_record_length(self
                          ):  # type: (...) -> int
        return self._record_length

    def get_nav_records(self,
                        gps_times   # type: np.ndarray
                        ):          # type: (...) -> np.ndarray
        return self._get_nav_records_native(gps_times)

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
