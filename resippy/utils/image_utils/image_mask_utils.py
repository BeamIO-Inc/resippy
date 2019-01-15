import resippy.utils.image_utils.image_utils as image_utils
import colorutils
import numpy as np
from typing import Union


def rgb_image_point_mask_to_pixel_locations(image,                  # type: ndarray
                                            hex_color_dict=None,    # type: dict
                                            ):                      # type: (...) -> Union[dict, tuple]
    # get image dims
    ny, nx, nbands = image_utils.get_image_ny_nx_nbands(image)
    # strip off alpha channel if the image has one
    if nbands == 4:
        image = image[:, :, 0:3]
    if hex_color_dict is not None:
        locs_dict = {}
        for key, val in list(hex_color_dict.items()):
            rgb_tuple = colorutils.hex_to_rgb(key)
            pixel_ys, pixel_xs = np.where(np.sum(image == rgb_tuple, axis=2) == 3)
            locs_dict[val] = (pixel_ys, pixel_xs)
        return locs_dict
    else:
        image = np.sum(image, axis=2)
        pixel_ys, pixel_xs = np.where(image > 0)
        return pixel_ys, pixel_xs
