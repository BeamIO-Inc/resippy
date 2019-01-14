from __future__ import division

import numpy as np
from numpy import ndarray
import warnings


def get_2d_cube_nx_ny_nbands(image_cube     # type: ndarray
                             ):             # type: (...) -> (int, int, int)
    cube_shape = image_cube.shape
    return cube_shape[1], cube_shape[0], cube_shape[2]


def is_2d_cube(image_cube   # type: ndarray
               ):           # type: (...) -> bool
    if len(np.shape(image_cube)) == 3:
        return True
    elif len(np.shape(image_cube)) == 2:
        return False
    else:
        warnings.warn("image data should either be 2 or 3 dimensional ndarrays, check your input data type")
        return False


def is_1d_cube(flattened_image_cube     # type: ndarray
               ):                       # type: (...) -> bool
    if len(np.shape(flattened_image_cube)) == 2:
        return True
    elif len(np.shape(flattened_image_cube)) == 3:
        return False
    else:
        warnings.warn("image data should either be 2 or 3 dimensional ndarrays, check your input data type")
        return False


def flatten_image_cube(image_cube       # type: ndarray
                       ):               # type: (...) -> ndarray
    ny = image_cube.shape[0]
    nx = image_cube.shape[1]
    n_bands = image_cube.shape[2]
    image_cube_1d = np.reshape(image_cube, (ny * nx, n_bands))
    return image_cube_1d


def unflatten_image_cube(image_cube_1d,     # type: ndarray
                         nx,                # type: int
                         ny                 # type: int
                         ):                 # type: (...) -> ndarray
    n_bands = np.shape(image_cube_1d)[1]
    hypercube = np.reshape(image_cube_1d, (ny, nx, n_bands))
    return hypercube
