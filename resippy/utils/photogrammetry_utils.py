from __future__ import division

from pyproj import transform
from functools import partial
from shapely.ops import transform as shapely_transform
import numpy as np
from numpy import ndarray
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Polygon

import resippy.photogrammetry.crs_defs as crs_defs


def reproject_geometry(geom,            # type: BaseGeometry
                       source_proj,     # type: str
                       dest_proj        # type: str
                       ):               # type: (...) -> BaseGeometry
    partial_transform = partial(transform, source_proj, dest_proj)
    return shapely_transform(partial_transform, geom)


# TODO: this might not be very accurate, create a better algorithm using local coordinate system
def shift_points_by_meters(meters_east,     # type: float
                           meters_north,    # type: float
                           source_proj,     # type: str
                           points_x,        # type: ndarray
                           points_y         # type: ndarray
                           ):               # type: (...) -> (ndarray, ndarray)
    points_x_meters, points_y_meters = transform(source_proj, crs_defs.PROJ_3857, points_x, points_y)
    shifted_x = points_x_meters + meters_east
    shifted_y = points_y_meters + meters_north
    shifted_x_native, shifted_y_native = transform(crs_defs.PROJ_3857, source_proj, shifted_x, shifted_y)
    return shifted_x_native, shifted_y_native


def world_poly_to_geo_t(world_polygon,      # type: Polygon
                        npix_x,             # type: int
                        npix_y              # type: int
                        ):                  # type: (...) -> list
    envelope = world_polygon.envelope
    minx, miny, maxx, maxy = envelope.bounds
    gsd_x = (maxx - minx)/(npix_x + 1)
    gsd_y = (maxy - miny)/(npix_y + 1)
    geot = [minx, gsd_x, 0, maxy, 0, -1*gsd_y]
    return geot


def create_ground_grid(min_x,   # type: float
                       max_x,   # type: float
                       min_y,   # type: float
                       max_y,   # type: float
                       npix_x,  # type: int
                       npix_y,  # type: int
                       ):       # type: (...) -> (ndarray, ndarray)
    ground_y_arr, ground_x_arr = np.mgrid[0:npix_y, 0:npix_x]
    ground_x_arr = ground_x_arr/npix_x*(max_x - min_x)
    ground_y_arr = (ground_y_arr - npix_y) * -1
    ground_y_arr = ground_y_arr/npix_y*(max_y - min_y)
    ground_x_arr = ground_x_arr + min_x
    ground_y_arr = ground_y_arr + min_y
    x_gsd = np.abs(ground_x_arr[0, 1] - ground_x_arr[0, 0])
    y_gsd = np.abs(ground_y_arr[0, 0] - ground_y_arr[1, 0])
    return ground_x_arr + x_gsd/2.0, ground_y_arr - y_gsd/2.0


def get_nx_ny_pixels_in_extent(extent,  # type: Polygon
                               gsd_x,  # type: float
                               gsd_y  # type: float
                               ):        # type: (...) -> (int, int)
    coords = extent.exterior.coords.xy

    x = coords[0]
    width = max(x) - min(x)
    nx = int(width / gsd_x)

    y = coords[1]
    height = max(y) - min(y)
    ny = int(height / gsd_y)

    return nx, ny
