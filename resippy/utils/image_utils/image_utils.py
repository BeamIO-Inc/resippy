from __future__ import division

from numpy import ndarray
import numpy as np
from typing import Union
from skimage import transform as sktransform
import gdal
import ogr


def create_uniform_image_data(nx,               # type: int
                              ny,               # type: int
                              nbands=1,         # type: int
                              values=128,       # type: Union[int, list]
                              dtype=np.uint8    # type: np.dtype
                              ):                # type: (...) -> ndarray
    image_data = np.zeros((ny, nx, nbands))
    if type(values) is int:
        image_data[:, :, :] = values
    else:
        for i, val in enumerate(values):
            image_data[:, :, i] = val
    return image_data.astype(dtype)


def grid_warp_image_band(image_to_warp,             # type: ndarray
                         image_x_coords,            # type: ndarray
                         image_y_coords,            # type: ndarray
                         nodata_val=0,              # type: int
                         interpolation='nearest'    # type: str
                         ):                         # type: (...) -> ndarray
    coords = np.array([image_y_coords, image_x_coords])
    if interpolation == 'nearest':
        order = 0
    elif interpolation == 'bilinear':
        order = 1
    elif interpolation == 'biquadratic':
        order = 2
    elif interpolation == 'bicubic':
        order = 3
    elif interpolation == 'biquartic':
        order = 4
    elif interpolation == 'biquintic':
        order = 5
    else:
        raise ValueError("Interpolation not supported: " + interpolation)

    warped_image = sktransform.warp(image_to_warp, coords, preserve_range=True,
                                    mode='constant', cval=nodata_val, order=order)
    return warped_image


def flatten_image_band(image_band       # type: ndarray
                       ):               # type: (...) -> ndarray
    ny = image_band.shape[0]
    nx = image_band.shape[1]
    image_band_1d = np.reshape(image_band, (ny * nx))
    return image_band_1d


def unflatten_image_band(image_band,        # type: ndarray
                         nx,                # type: int
                         ny                 # type: int
                         ):                 # type: (...) -> ndarray
    return np.reshape(image_band, (ny, nx))


def create_pixel_grid(nx_pixels,        # type: int
                      ny_pixels,        # type: int
                      scale_factor=1    # type: float
                      ):                # type: (...) -> (ndarray, ndarray)
    x = np.arange(0, nx_pixels, (1.0 / scale_factor))
    y = np.arange(0, ny_pixels, (1.0 / scale_factor))
    xx, yy = np.meshgrid(x, y, sparse=False)
    return xx, yy


# TODO needs testing
def gdal_grid_image_band(image_to_warp,     # type: ndarray
                         output_fname,      # type: str
                         image_x_coords,    # type: ndarray
                         image_y_coords,    # type: ndarray
                         npix_x,            # type: int
                         npix_y,            # type: int
                         projection_wkt,    # type: str
                         nodata_val=0       # type: int
                         ):                 # type:

    fileformat = "MEMORY"
    driver = ogr.GetDriverByName(fileformat)
    ys, xs = image_to_warp.size

    datasource = driver.CreateDataSource('memData')
    datasource.SetProjection(projection_wkt)

    layer = datasource.CreateLayer('image', geom_type=ogr.wkbPoint)
    layer.CreateField(ogr.FieldDefn("Z", ogr.OFTReal))

    for i in range(xs):
        for j in range(ys):

            feature = ogr.Feature(layer.GetLayerDefn())
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(image_x_coords[j,i], image_y_coords[j,i])
            feature.SetGeometry(point)

            feature.SetField("Z", image_to_warp[j,i])

            layer.CreateFeature(feature)
            feature = None

    dst_fname = output_fname

    ops = gdal.GridOptions(width=npix_x, height=npix_y, noData=nodata_val)

    dst_dataset = gdal.Grid(dst_fname, datasource, options=ops)
    src_dataset = None
    dst_dataset = None

    print("done")


def is_grayscale(image_2d       # type: ndarray
                 ):             # type: (...) -> bool
    return len(image_2d.shape) == 2


def get_image_ny_nx_nbands(image_2d     # type: ndarray
                           ):           # type: (...) -> (int, int, int)
    is_input_grayscale = is_grayscale(image_2d)
    ny = image_2d.shape[0]
    nx = image_2d.shape[1]
    if is_input_grayscale:
        nbands = 1
    else:
        nbands = image_2d.shape[2]
    return ny, nx, nbands

