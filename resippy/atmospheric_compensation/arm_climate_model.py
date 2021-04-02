import numpy
from numpy import ndarray
from PIL import Image

import matplotlib.pyplot as plt


class ArmClimateModel:
    def __init__(self):
        self._cloud_mask  # type: ndarray

    @classmethod
    def from_image_file(cls, image_path):
        image = numpy.array(Image.open(image_path))
        ny, nx, nbands = image.shape
        horizontal_cross_section = image[int(ny/2), :][:, 0]
        horizontal_start_index = numpy.argmax(horizontal_cross_section < 255)
        horizontal_tmp = horizontal_cross_section[::-1]
        horizontal_end_index = len(horizontal_tmp) - numpy.argmax(horizontal_tmp < 255) - 1
        image_subsection = image[:, horizontal_start_index:horizontal_end_index]

        vertical_cross_section = image_subsection[:, 0][:, 0]
        vertical_start_index = numpy.argmax(vertical_cross_section < 255)
        vertical_tmp = vertical_cross_section[::-1]
        vertical_end_index = len(vertical_tmp) - numpy.argmax(vertical_tmp < 255) - 1
        image_subsection = image_subsection[vertical_start_index: vertical_end_index, :]

        return cls.from_numpy_array(image_subsection)

    @classmethod
    def from_numpy_array(cls,
                         array,
                         ):
        ny, nx, nbands = numpy.shape(array)
        center_y, center_x = ny/2, nx/2
        y_pixel_locs = numpy.repeat(numpy.arange(0, ny), nx).reshape(ny, nx)
        x_pixel_locs = numpy.tile(numpy.arange(0, nx), ny).reshape(ny, nx)
        stop = 1