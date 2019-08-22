from datetime import datetime, timedelta
from bisect import bisect_right
import math

# table of UTC leap second insertions since 1980 (http://hpiers.obspm.fr/eop-pc/index.php?index=TAI-UTC_tab&lang=en)
_LEAPSECONDS = [
    (datetime(1980, 1, 1).timestamp(), timedelta(seconds=19)),
    (datetime(1981, 7, 1).timestamp(), timedelta(seconds=20)),
    (datetime(1982, 7, 1).timestamp(), timedelta(seconds=21)),
    (datetime(1983, 7, 1).timestamp(), timedelta(seconds=22)),
    (datetime(1985, 7, 1).timestamp(), timedelta(seconds=23)),
    (datetime(1988, 1, 1).timestamp(), timedelta(seconds=24)),
    (datetime(1990, 1, 1).timestamp(), timedelta(seconds=25)),
    (datetime(1991, 1, 1).timestamp(), timedelta(seconds=26)),
    (datetime(1992, 7, 1).timestamp(), timedelta(seconds=27)),
    (datetime(1993, 7, 1).timestamp(), timedelta(seconds=28)),
    (datetime(1994, 7, 1).timestamp(), timedelta(seconds=29)),
    (datetime(1996, 1, 1).timestamp(), timedelta(seconds=30)),
    (datetime(1997, 7, 1).timestamp(), timedelta(seconds=31)),
    (datetime(1999, 1, 1).timestamp(), timedelta(seconds=32)),
    (datetime(2006, 1, 1).timestamp(), timedelta(seconds=33)),
    (datetime(2009, 1, 1).timestamp(), timedelta(seconds=34)),
    (datetime(2012, 7, 1).timestamp(), timedelta(seconds=35)),
    (datetime(2015, 7, 1).timestamp(), timedelta(seconds=36)),
    (datetime(2017, 1, 1).timestamp(), timedelta(seconds=37))
]

# reference epoch for GPS and UTC time
_GPS_EPOCH = datetime(1980, 1, 6)
_UNIX_EPOCH = datetime(1970, 1, 1)

# other constants
_SECONDS_IN_MINUTE = 60
_MINUTES_IN_HOUR = 60
_HOURS_IN_DAY = 24
_DAYS_IN_WEEK = 7
_SECONDS_IN_HOUR = _SECONDS_IN_MINUTE * _MINUTES_IN_HOUR
_SECONDS_IN_DAY = _SECONDS_IN_HOUR * _HOURS_IN_DAY
_SECONDS_IN_WEEK = _DAYS_IN_WEEK * _SECONDS_IN_DAY


def _get_leapseconds(timestamp  # type: float
                     ):         # type: (...) -> timedelta
    leapsecond_dates = [leapsecond[0] for leapsecond in _LEAPSECONDS]
    index = bisect_right(leapsecond_dates, timestamp) - 1
    return _LEAPSECONDS[index][1]


def _get_gps_leapsecond_offset(timestamp    # type: float
                               ):           # type: (...) -> timedelta
    return _get_leapseconds(timestamp) - _get_leapseconds(_GPS_EPOCH.timestamp())


def utc_timestamp_to_gps_timestamp(utc_timestamp    # type: float
                                   ):               # type: (...) -> float
    return utc_timestamp - (_GPS_EPOCH.timestamp() - _UNIX_EPOCH.timestamp()) + \
           _get_gps_leapsecond_offset(utc_timestamp).total_seconds()


def gps_timestamp_to_utc_timestamp(gps_timestamp    # type: float
                                   ):               # type: (...) -> float
    utc_timestamp_with_leap = gps_timestamp + (_GPS_EPOCH.timestamp() - _UNIX_EPOCH.timestamp())
    return utc_timestamp_with_leap - _get_gps_leapsecond_offset(utc_timestamp_with_leap).total_seconds()


def utc_timestamp_to_gps_week_and_seconds(utc_timestamp     # type: float
                                          ):                # type: (...) -> (float, float)
    gps_timestamp = utc_timestamp_to_gps_timestamp(utc_timestamp)

    weeks = gps_timestamp / _SECONDS_IN_WEEK
    seconds_in_week = (weeks % 1) * _DAYS_IN_WEEK * _SECONDS_IN_DAY

    return math.floor(weeks), seconds_in_week
