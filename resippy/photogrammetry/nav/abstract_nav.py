import abc
from six import add_metaclass


@add_metaclass(abc.ABCMeta)
class AbstractNav:

    def __init__(self):
        self._num_records = 0
        self._record_length = 0
        self._nav_data = None

    @abc.abstractmethod
    def get_nav_record(self,
                       gps_time     # type: float
                       ):           # type: (...) -> dict
        pass

    def _gps_time_in_range(self,
                           gps_time     # type: float
                           ):           # type: (...) -> bool
        if self._nav_data['gps_time'][0] <= gps_time <= self._nav_data['gps_time'][-1]:
            return True

        return False

    def get_num_records(self
                        ):  # type: (...) -> int
        return self._num_records

    def get_record_length(self
                          ):    # type: (...) -> int
        return self._record_length
