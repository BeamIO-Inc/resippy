from __future__ import division

from shapely.geometry import MultiPoint
from shapely.geometry import Polygon


def bounds_2_shapely_polygon(min_x,     # type: float
                             max_x,     # type: float
                             max_y,     # type: float
                             min_y      # type: float
                             ):         # type: (...) -> Polygon
    points = [(min_x, max_y), (max_x, min_y)]
    envelope = MultiPoint(points).envelope
    return envelope


def deg_min_sec_to_decimal_degrees(lon_deg,         # type: int
                                   lon_min,         # type: int
                                   lon_sec,         # type: float
                                   lon_E_or_W,      # type: str
                                   lat_deg,         # type: int
                                   lat_min,         # type: int
                                   lat_sec,         # type: float
                                   lat_N_or_S       # type: str
                                   ):               # type: (...) -> tuple
    e_or_w = lon_E_or_W.lower()
    n_or_s = lat_N_or_S.lower()

    lon_sign = 1
    if e_or_w == 'w':
        lon_sign = -1

    lat_sign = 1
    if n_or_s == 's':
        lat_sign = -1

    decimal_lon = lon_sign * (int(lon_deg) + float(lon_min) / 60 + float(lon_sec) / 3600)
    decimal_lat = lat_sign * (int(lat_deg) + float(lat_min) / 60 + float(lat_sec) / 3600)
    return decimal_lon, decimal_lat
