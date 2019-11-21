from __future__ import division

import numpy as np
from numpy import ndarray
from typing import Union
import os
import imageio
import resippy.utils.image_utils.image_utils as image_utils


def get_nchips(image_chips,     # type: ndarray
               ): # type: (...) -> int
    return image_chips.shape[0]


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


# TODO support for chipping outside of image bounds is not yet supported.
# TODO Indices will be adjusted to keep within image bounds.
def chip_images_by_pixel_upper_lefts(input_image,                   # type: ndarray
                                     pixel_y_ul_list,               # type: list
                                     pixel_x_ul_list,               # type: list
                                     chip_ny_pixels=256,            # type: int
                                     chip_nx_pixels=256,            # type: int
                                     bands=None,                    # type: Union[list, int]
                                     keep_within_image_bounds=True,  # type: bool
                                     atk_chain_ledger=None
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
    n_pix = len(pixel_x_ul_list)
    for cnt, dat in enumerate(zip(pixel_y_ul_list, pixel_x_ul_list)):
        y, x = dat
        all_chips[chip_counter, :, :, :] = \
            input_image[y: y + chip_ny_pixels, x: x + chip_nx_pixels, bands]
        chip_counter += 1
        upper_lefts.append((y, x))
        per = round(float(cnt) / float(n_pix) * 100.0)
        if per % 5 == 0:
            if atk_chain_ledger is not None:
                atk_chain_ledger.set_status('generating chips', per)

    if is_output_grayscale:
        all_chips = np.reshape(all_chips, (n_chips, chip_ny_pixels, chip_nx_pixels))

    return all_chips, upper_lefts


def chip_images_by_pixel_centers(input_image,                   # type: ndarray
                                 pixel_y_center_list,               # type: list
                                 pixel_x_center_list,               # type: list
                                 chip_ny_pixels=256,            # type: int
                                 chip_nx_pixels=256,            # type: int
                                 bands=None,                    # type: Union[list, int]
                                 keep_within_image_bounds=True,  # type: bool
                                 atk_chain_ledger=None,         #  type: AlgorithmChain.ChainLedger
                                 ):                             # type: (...) -> (ndarray, list)
    pixel_x_center_list = np.array(pixel_x_center_list)
    pixel_y_center_list = np.array(pixel_y_center_list)
    pixel_x_ul_list = (pixel_x_center_list - chip_nx_pixels/2.0).astype(int)
    pixel_y_ul_list = (pixel_y_center_list - chip_ny_pixels/2.0).astype(int)
    return chip_images_by_pixel_upper_lefts(
        input_image, pixel_y_ul_list=pixel_y_ul_list, pixel_x_ul_list=pixel_x_ul_list,
        chip_ny_pixels=chip_ny_pixels, chip_nx_pixels=chip_nx_pixels,
        bands=bands, keep_within_image_bounds=keep_within_image_bounds, atk_chain_ledger=atk_chain_ledger)


def write_chips_to_disk(image_chips,            # type: ndarray
                        output_dir,             # type: str
                        base_chip_fname=None,   # type: str
                        fnames_list=None,       # type: list
                        output_chip_ny=None,    # type: int
                        output_chip_nx=None,    # type: int
                        remove_alpha=True,      # type: bool
                        output_format="png",     # type: str
                        atk_chain_ledger=None
                        ):  # type: (...) -> None
    if base_chip_fname is None:
        base_chip_fname = os.path.basename(output_dir)
    n_chips = get_nchips(image_chips)
    for i in range(n_chips):
        chip = image_chips[i, :, :, :]
        if output_chip_ny is not None:
            chip = image_utils.resize_image(chip, output_chip_ny, output_chip_nx)
        if remove_alpha is True and chip.shape[-1] == 4:
            chip = chip[:, :, 0:3]
        chip_fname = base_chip_fname + "_" + str(i).zfill(8)
        if fnames_list is not None:
            chip_fname = fnames_list[i]
        chip_fname = chip_fname + "." + output_format.replace(".", "")
        chip_fullpath = os.path.join(output_dir, chip_fname)
        imageio.imsave(chip_fullpath, chip)

        per = round(float(i) / float(n_chips) * 100.0)
        if per % 5 == 0:
            if atk_chain_ledger is not None:
                atk_chain_ledger.set_status('writing: ' + chip_fullpath, per)
