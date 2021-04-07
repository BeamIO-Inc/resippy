import numpy
from numpy import ndarray
from typing import Union
import numbers


def az_el_r_to_xyz(azimuths_angles,  # type: Union[ndarray, float]
                   elevation_angles,  # type: Union[ndarray, float]
                   radius=1.0,  # type: Union[ndarray, float]
                   combine_arrays=False,  # type: bool
                   ):
    zeniths = numpy.pi / 2 - elevation_angles
    x = radius * numpy.sin(zeniths) * numpy.cos(azimuths_angles)
    y = radius * numpy.sin(zeniths) * numpy.sin(azimuths_angles)
    z = radius * numpy.cos(zeniths)
    if combine_arrays:
        if isinstance(azimuths_angles, numbers.Number):
            arr_len = 1
        else:
            arr_len = len(zeniths)
        xyz = numpy.zeros((arr_len, 3))
        xyz[:, 0] = x
        xyz[:, 1] = y
        xyz[:, 2] = z
        return xyz
    else:
        return x, y, z


def az_el_to_xy_plane(azimuth_angles,
                      elevation_angles,
                      z_location,
                      hemisphere_xyz_loc=[0, 0, 0]
                      ):
    height = z_location - hemisphere_xyz_loc[2]
    x, y, z = az_el_r_to_xyz(azimuth_angles, elevation_angles, 1.0)
    new_x = x / z * height + hemisphere_xyz_loc[0]
    new_y = y / z * height + hemisphere_xyz_loc[1]
    z = numpy.zeros_like(z) + z_location
    return new_x, new_y, z


def xyz_to_az_el_radius(x,  # type: Union[ndarray, float]
                        y,  # type: Union[ndarray, float]
                        z,  # type: Union[ndarray, float]
                        combine_arrays=False,  # type: bool
                        ):
    xy = numpy.power(x, 2) + numpy.power(y, 2)
    el = numpy.arctan2(z, numpy.sqrt(xy))
    az = numpy.arctan2(y, x)
    az[numpy.where(az < 0)] = az[numpy.where(az < 0)] + 2 * numpy.pi
    radius = numpy.sqrt(xy + numpy.power(z, 2))
    if combine_arrays:
        if isinstance(x, numbers.Number):
            arr_len = 1
        else:
            arr_len = len(x)
        az_el_r = numpy.zeros((arr_len, 3))
        az_el_r[:, 0] = az
        az_el_r[:, 1] = el
        az_el_r[:, 2] = radius
        return az_el_r
    else:
        return az, el, radius
