from __future__ import division

import numpy as np
from numpy import ndarray
import warnings


def get_2d_cube_nx_ny_nbands(image_cube     # type: ndarray
                             ):             # type: (...) -> (int, int, int)
    # TODO: This should probably be renamed and re-implemented to be ny, nx, nbands, for consistency across the project
    cube_shape = image_cube.shape
    return cube_shape[1], cube_shape[0], cube_shape[2]


def is_2d_cube(image_cube   # type: ndarray
               ):           # type: (...) -> bool
    """
    This method specifies whether an image cube is a 2d or a 1d cube.
    A 2d cube has dimensions (y pixels, x pixels, spectral), and a 1d cube has dimensions (samples, spectral)
    :param image_cube: input image cube
    :return: True if the cube is 2d, false otherwise (with a warning if the cube is neither 1d or 2d)
    """
    if len(np.shape(image_cube)) == 3:
        return True
    elif len(np.shape(image_cube)) == 2:
        return False
    else:
        warnings.warn("image data should either be 2 or 3 dimensional ndarrays, check your input data type")
        return False


def is_1d_cube(flattened_image_cube     # type: ndarray
               ):                       # type: (...) -> bool
    """
    Specifies whether an image cube is 1d (has dimensions of (samples, nbands), rather than (ny, nx, nbands))
    :param flattened_image_cube: input image cube
    :return: True is cube is 1d, False otherwise (with a warning if the cube is neither 1d or 2d)
    """
    if len(np.shape(flattened_image_cube)) == 2:
        return True
    elif len(np.shape(flattened_image_cube)) == 3:
        return False
    else:
        warnings.warn("image data should either be 2 or 3 dimensional ndarrays, check your input data type")
        return False


def flatten_image_cube(image_cube       # type: ndarray
                       ):               # type: (...) -> ndarray
    """
    Takes a 2d image cube and flattens it to a 1d cube.
    :param image_cube: 2d image cube of dimensions (ny, nx, nbands)
    :return: flattened cube of dimensions (ny * nx, nbands)
    """
    ny = image_cube.shape[0]
    nx = image_cube.shape[1]
    n_bands = image_cube.shape[2]
    image_cube_1d = np.reshape(image_cube, (ny * nx, n_bands))
    return image_cube_1d


def unflatten_image_cube(image_cube_1d,     # type: ndarray
                         nx,                # type: int
                         ny                 # type: int
                         ):                 # type: (...) -> ndarray
    """
    Takes a 1d image cube and creates a 2d cube
    :param image_cube_1d: Input image cube of dimensions (ny * nx, nbands)
    :param nx: number of x pixels for output 2d cube
    :param ny: number of y pixels for output 2d cube
    :return: unflattened image cube of dimensions (ny, nx, nbands)
    """
    n_bands = np.shape(image_cube_1d)[1]
    hypercube = np.reshape(image_cube_1d, (ny, nx, n_bands))
    return hypercube
