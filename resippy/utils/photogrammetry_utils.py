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


def create_M_matrix(omega_radians, phi_radians, kappa_radians):

    m_matrix = np.zeros((3,3))

    m11 = np.cos(phi_radians) * np.cos(kappa_radians)
    m12 = np.cos(omega_radians) * np.sin(kappa_radians) + np.sin(omega_radians) * np.sin(phi_radians) * np.cos(
        kappa_radians)
    m13 = np.sin(omega_radians) * np.sin(kappa_radians) - np.cos(omega_radians) * np.sin(phi_radians) * np.cos(
        kappa_radians)

    m21 = -1.0 * np.cos(phi_radians) * np.sin(kappa_radians)
    m22 = np.cos(omega_radians) * np.cos(kappa_radians) - np.sin(omega_radians) * np.sin(phi_radians) * np.sin(
        kappa_radians)
    m23 = np.sin(omega_radians) * np.cos(kappa_radians) + np.cos(omega_radians) * np.sin(phi_radians) * np.sin(
        kappa_radians)

    m31 = np.sin(phi_radians)
    m32 = -1.0 * np.sin(omega_radians) * np.cos(phi_radians)
    m33 = np.cos(omega_radians) * np.cos(phi_radians)

    m_matrix[0, 0] = m11
    m_matrix[0, 1] = m12
    m_matrix[0, 2] = m13

    m_matrix[1, 0] = m21
    m_matrix[1, 1] = m22
    m_matrix[1, 2] = m23

    m_matrix[2, 0] = m31
    m_matrix[2, 1] = m32
    m_matrix[2, 2] = m33

    return m_matrix


def solve_for_omega_phi_kappa(m_matrix,  # type: ndarray
                              ):  # type: (...) -> tuple

    # m11 = cos(phi) * cos(kappa)
    # m31 = sin(phi)
    # m33 = cos(omega) * cos(phi)
    # m32 = -1.0 * np.sin(omega) * np.cos(phi)
    # m21 = -1.0 * np.cos(phi) * np.sin(kappa)

    # 3 equations, 3 unknowns
    m11 = m_matrix[0, 0]
    m31 = m_matrix[2, 0]
    m33 = m_matrix[2, 2]
    m32 = m_matrix[2, 1]
    m21 = m_matrix[1, 0]

    phi = np.arcsin(m31)
    cos_phi = np.cos(phi)
    kappa = np.arccos(m11 / cos_phi)
    omega = np.arccos(m33 / cos_phi)
    if m32 > 0:
        omega = -omega
    if m21 > 0:
        kappa = -kappa

    return omega, phi, kappa
