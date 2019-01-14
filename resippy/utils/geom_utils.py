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
