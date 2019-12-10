from __future__ import division

from numpy import ndarray
import numpy as np
from typing import Union
from skimage import transform as sktransform
import gdal
import ogr
import seaborn
from seaborn.palettes import _ColorPalette
from PIL import Image

def create_uniform_image_data(nx,  # type: int
                              ny,  # type: int
                              nbands=1,  # type: int
                              values=128,  # type: Union[int, list]
                              dtype=np.uint8  # type: np.dtype
                              ):  # type: (...) -> ndarray
    image_data = np.zeros((ny, nx, nbands))
    if type(values) is int:
        image_data[:, :, :] = values
    else:
        for i, val in enumerate(values):
            image_data[:, :, i] = val
    return image_data.astype(dtype)


def grid_warp_image_band(image_to_warp,  # type: ndarray
                         image_x_coords,  # type: ndarray
                         image_y_coords,  # type: ndarray
                         nodata_val=0,  # type: int
                         interpolation='nearest'  # type: str
                         ):  # type: (...) -> ndarray
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


def flatten_image_band(image_band  # type: ndarray
                       ):  # type: (...) -> ndarray
    ny = image_band.shape[0]
    nx = image_band.shape[1]
    image_band_1d = np.reshape(image_band, (ny * nx))
    return image_band_1d


def unflatten_image_band(image_band,  # type: ndarray
                         nx,  # type: int
                         ny  # type: int
                         ):  # type: (...) -> ndarray
    return np.reshape(image_band, (ny, nx))


def create_pixel_grid(nx_pixels,  # type: int
                      ny_pixels,  # type: int
                      scale_factor=1  # type: float
                      ):  # type: (...) -> (ndarray, ndarray)
    x = np.arange(0, nx_pixels, (1.0 / scale_factor))
    y = np.arange(0, ny_pixels, (1.0 / scale_factor))
    xx, yy = np.meshgrid(x, y, sparse=False)
    return xx, yy


# TODO needs testing
def gdal_grid_image_band(image_to_warp,  # type: ndarray
                         output_fname,  # type: str
                         image_x_coords,  # type: ndarray
                         image_y_coords,  # type: ndarray
                         npix_x,  # type: int
                         npix_y,  # type: int
                         projection_wkt,  # type: str
                         nodata_val=0  # type: int
                         ):  # type:

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
            point.AddPoint(image_x_coords[j, i], image_y_coords[j, i])
            feature.SetGeometry(point)

            feature.SetField("Z", image_to_warp[j, i])

            layer.CreateFeature(feature)
            feature = None

    dst_fname = output_fname

    ops = gdal.GridOptions(width=npix_x, height=npix_y, noData=nodata_val)

    dst_dataset = gdal.Grid(dst_fname, datasource, options=ops)
    src_dataset = None
    dst_dataset = None

    print("done")


def is_grayscale(image_2d  # type: ndarray
                 ):  # type: (...) -> bool
    return len(image_2d.shape) == 2


def get_image_ny_nx_nbands(image_2d  # type: ndarray
                           ):  # type: (...) -> (int, int, int)
    is_input_grayscale = is_grayscale(image_2d)
    ny = image_2d.shape[0]
    nx = image_2d.shape[1]
    if is_input_grayscale:
        nbands = 1
    else:
        nbands = image_2d.shape[2]
    return ny, nx, nbands


def create_detection_image_from_scores(score_values, upper_left_yx_tuples, chip_size_x, chip_size_y):
    ny = upper_left_yx_tuples[:, 0].max() + chip_size_y
    nx = upper_left_yx_tuples[:, 1].max() + chip_size_x
    detection_image = np.zeros((ny, nx))
    score_sorted_indices = np.argsort(score_values)
    for score_index in score_sorted_indices:
        ul_yx = upper_left_yx_tuples[score_index]
        score = score_values[score_index]
        detection_image[ul_yx[0]:ul_yx[0] + chip_size_y, ul_yx[1]:ul_yx[1] + chip_size_x] = score
    return detection_image


# TODO: add support for discrete color steps, rather than just continous
def apply_colormap_to_grayscale_image(grayscale_image,  # type: ndarray
                                      color_palette=None,  # type: Union[_ColorPalette, ndarray]
                                      continuous=True,  # type: bool
                                      n_bins=None,  # type: int
                                      min_clip=None,  # type: float
                                      max_clip=None,  # type: float
                                      output_dtype=np.uint8,
                                      nodata=None  # type: Union[int, float]
                                      ):  # type: (...) -> ndarray
    if nodata is not None:
        good_inds = grayscale_image != nodata
    if color_palette is None:
        color_palette = seaborn.color_palette("GnBu_r")
    if min_clip is None:
        if nodata is not None:
            min_clip = np.nanmin(grayscale_image[good_inds])
        else:
            min_clip = np.nanmin(grayscale_image)
    if max_clip is None:
        if nodata is not None:
            max_clip = np.nanmax(grayscale_image[good_inds])
        else:
            max_clip = np.nanmax(grayscale_image)
    color_palette = np.asarray(color_palette)

    if n_bins is not None:
        tmp_grayscale_image = np.linspace(min_clip, max_clip, n_bins)
        tmp_grayscale_image = np.expand_dims(tmp_grayscale_image, 0)
        new_color_palette = apply_colormap_to_grayscale_image(tmp_grayscale_image,
                                                              color_palette,
                                                              continuous=True,
                                                              output_dtype=np.float64)
        new_color_palette = np.squeeze(new_color_palette / 255.0)
        color_palette = new_color_palette

    if n_bins is None:
        n_bins = np.shape(color_palette)[0]

    if continuous is not True:
        max_clip = min_clip + (max_clip - min_clip) * (n_bins - 1) / n_bins

    palette_indices = np.linspace(min_clip, max_clip, n_bins)
    image_low_index_map = np.zeros(grayscale_image.shape, dtype=np.int8)
    image_high_index_map = np.zeros(grayscale_image.shape, dtype=np.int8)
    image_low_palette_index = np.zeros(grayscale_image.shape)
    image_high_palette_index = np.zeros(grayscale_image.shape)

    palette_reds_low = np.zeros(grayscale_image.shape)
    palette_reds_high = np.zeros(grayscale_image.shape)
    palette_greens_low = np.zeros(grayscale_image.shape)
    palette_greens_high = np.zeros(grayscale_image.shape)
    palette_blues_low = np.zeros(grayscale_image.shape)
    palette_blues_high = np.zeros(grayscale_image.shape)

    for i in range(n_bins - 1):
        bin_indices = np.logical_and(grayscale_image >= palette_indices[i], grayscale_image <= palette_indices[i + 1])
        image_low_index_map[bin_indices] = i
        image_high_index_map[bin_indices] = i + 1
        image_low_palette_index[bin_indices] = palette_indices[i]
        image_high_palette_index[bin_indices] = palette_indices[i + 1]
    palette_index_weights_high = (grayscale_image - image_low_palette_index) / \
                                 (image_high_palette_index - image_low_palette_index)
    palette_index_weights_low = 1.0 - palette_index_weights_high

    for i in range(n_bins - 1):
        palette_reds_low[np.where(image_low_index_map == i)] = color_palette[i, 0]
        palette_reds_high[np.where(image_low_index_map == i)] = color_palette[i + 1, 0]
        palette_greens_low[np.where(image_low_index_map == i)] = color_palette[i, 1]
        palette_greens_high[np.where(image_low_index_map == i)] = color_palette[i + 1, 1]
        palette_blues_low[np.where(image_low_index_map == i)] = color_palette[i, 2]
        palette_blues_high[np.where(image_low_index_map == i)] = color_palette[i + 1, 2]

    ny, nx, bands = get_image_ny_nx_nbands(grayscale_image)
    colormapped_image = np.zeros((ny, nx, 3))

    if continuous is True:
        colormapped_image[:, :, 0] = palette_reds_low * palette_index_weights_low + \
                                     palette_reds_high * palette_index_weights_high
        colormapped_image[:, :, 1] = palette_greens_low * palette_index_weights_low + \
                                     palette_greens_high * palette_index_weights_high
        colormapped_image[:, :, 2] = palette_blues_low * palette_index_weights_low + \
                                     palette_blues_high * palette_index_weights_high
    else:
        colormapped_image[:, :, 0] = palette_reds_low
        colormapped_image[:, :, 1] = palette_greens_low
        colormapped_image[:, :, 2] = palette_blues_low

    colormapped_image[grayscale_image < min_clip, :] = color_palette[0, :]
    colormapped_image[grayscale_image > max_clip, :] = color_palette[-1, :]

    colormapped_image = colormapped_image * 255.0
    colormapped_image = np.asarray(colormapped_image, dtype=output_dtype)

    return colormapped_image


def blend_images(image_1,  # type: ndarray
                 image_2,  # type: ndarray
                 image_1_percent=0.5  # type: float
                 ):  # type: (...) -> ndarray
    return image_1 * image_1_percent + image_2 * (1 - image_1_percent)


def resize_image(image_to_resize,           # type: ndarray
                 new_ny,                    # type: int
                 new_nx,                    # type: int
                 ):                         # type: (...) -> ndarray
    pil_image = Image.fromarray(image_to_resize)
    resized_pil_image = Image.Image.resize(pil_image, (new_nx, new_ny))
    resized_numpy_image = np.array(resized_pil_image)
    return resized_numpy_image
