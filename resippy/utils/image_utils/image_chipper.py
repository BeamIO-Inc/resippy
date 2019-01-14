from __future__ import division

import numpy as np
from numpy import ndarray
from typing import Union


def chip_entire_image_to_memory(input_image,                    # type: ndarray
                                chip_ny_pixels=256,             # type: int
                                chip_nx_pixels=256,             # type: int
                                npix_overlap_y=0,               # type: int
                                npix_overlap_x=0,               # type: int
                                bands=None,                     # type: Union[list, int]
                                keep_within_image_bounds=True   # type: bool
                                ):                              # type: (...) -> (ndarray, list)

    is_input_grayscale = len(input_image.shape) == 2
    if is_input_grayscale:
        ny = input_image.shape[0]
        nx = input_image.shape[1]
        input_image = np.reshape(input_image, (ny, nx, 1))
    ny, nx, nbands = input_image.shape
    y_idxs = np.arange(0, ny, chip_ny_pixels - npix_overlap_y)
    x_idxs = np.arange(0, nx, chip_nx_pixels - npix_overlap_x)
    y_idxs[np.where(y_idxs >= ny - chip_ny_pixels)] = ny - chip_ny_pixels
    x_idxs[np.where(x_idxs >= nx - chip_nx_pixels)] = nx - chip_nx_pixels
    y_idxs = sorted(list(set(y_idxs)))
    x_idxs = sorted(list(set(x_idxs)))

    y_ul_list = []
    x_ul_list = []

    for y_index in y_idxs:
        for x_index in x_idxs:
            y_ul_list.append(y_index)
            x_ul_list.append(x_index)

    return chip_images_by_pixel_upper_lefts(input_image,
                                            pixel_y_ul_list=y_ul_list,
                                            pixel_x_ul_list=x_ul_list,
                                            chip_ny_pixels=chip_ny_pixels,
                                            chip_nx_pixels=chip_nx_pixels,
                                            bands=bands,
                                            keep_within_image_bounds=keep_within_image_bounds)


# TODO support for chipping outside of image bounds is not yet supported.  Indices will be adjusted to keep within image bounds.
def chip_images_by_pixel_upper_lefts(input_image,                   # type: ndarray
                                     pixel_y_ul_list,               # type: list
                                     pixel_x_ul_list,               # type: list
                                     chip_ny_pixels=256,            # type: int
                                     chip_nx_pixels=256,            # type: int
                                     bands=None,                    # type: Union[list, int]
                                     keep_within_image_bounds=True  # type: bool
                                     ):                             # type: (...) -> (ndarray, list)

    if type(bands) is type(0):
        bands = [bands]

    pixel_x_ul_list = np.array(pixel_x_ul_list)
    pixel_y_ul_list = np.array(pixel_y_ul_list)
    is_input_grayscale = len(input_image.shape) == 2
    if is_input_grayscale:
        ny = input_image.shape[0]
        nx = input_image.shape[1]
        input_image = np.reshape(input_image, (ny, nx, 1))
    ny, nx, nbands = input_image.shape

    if keep_within_image_bounds:
        pixel_x_ul_list[np.where(pixel_x_ul_list > nx - chip_nx_pixels)] = nx - chip_nx_pixels
        pixel_y_ul_list[np.where(pixel_y_ul_list > ny - chip_ny_pixels)] = ny - chip_ny_pixels

        pixel_x_ul_list[np.where(pixel_x_ul_list < 0)] = 0
        pixel_y_ul_list[np.where(pixel_y_ul_list < 0)] = 0

    n_chips = len(pixel_y_ul_list)

    if bands is None:
        bands = np.arange(0, nbands).astype(np.int)
    else:
        nbands = len(bands)

    is_output_grayscale = is_input_grayscale or len(bands) == 1
    all_chips = np.zeros((n_chips, chip_ny_pixels, chip_nx_pixels, nbands), dtype=input_image.dtype)

    chip_counter = 0
    upper_lefts = []
    for y, x in zip(pixel_y_ul_list, pixel_x_ul_list):
        all_chips[chip_counter, :, :, :] = \
            input_image[y: y + chip_ny_pixels, x: x + chip_nx_pixels, bands]
        chip_counter += 1
        upper_lefts.append((y, x))

    if is_output_grayscale:
        all_chips = np.reshape(all_chips, (n_chips, chip_ny_pixels, chip_nx_pixels))

    return all_chips, upper_lefts


def chip_images_by_pixel_centers(input_image,                   # type: ndarray
                                 pixel_y_center_list,               # type: list
                                 pixel_x_center_list,               # type: list
                                 chip_ny_pixels=256,            # type: int
                                 chip_nx_pixels=256,            # type: int
                                 bands=None,                    # type: Union[list, int]
                                 keep_within_image_bounds=True  # type: bool
                                 ):                             # type: (...) -> (ndarray, list)
    pixel_x_center_list = np.array(pixel_x_center_list)
    pixel_y_center_list = np.array(pixel_y_center_list)
    pixel_x_ul_list = (pixel_x_center_list - chip_nx_pixels/2.0).astype(int)
    pixel_y_ul_list = (pixel_y_center_list - chip_ny_pixels/2.0).astype(int)
    return chip_images_by_pixel_upper_lefts(
        input_image, pixel_y_ul_list=pixel_y_ul_list, pixel_x_ul_list=pixel_x_ul_list,
        chip_ny_pixels=chip_ny_pixels, chip_nx_pixels=chip_nx_pixels,
        bands=bands, keep_within_image_bounds=keep_within_image_bounds)
