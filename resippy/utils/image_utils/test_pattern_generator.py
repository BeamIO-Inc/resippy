import numpy
import matplotlib.pyplot as plt


def gradient_image(ny, nx, ul_val=0, mid_val=0.5, br_val=1):
    total_length = nx*ny
    first_part = numpy.linspace(ul_val, mid_val, int(total_length/2))
    last_part = numpy.linspace(mid_val, br_val, int(total_length/2))
    gradient_1d_arr = numpy.zeros(total_length)
    gradient_1d_arr[0:int(total_length/2)] = first_part
    gradient_1d_arr[int(total_length/2)-1:-1] = last_part
    pattern_image = numpy.reshape(gradient_1d_arr, (ny, nx))
    return pattern_image


def red_to_blue(ny, nx):
    red_pattern = gradient_image(ny, nx, ul_val=255, mid_val=128, br_val=0)
    green_pattern = numpy.zeros_like(red_pattern)
    blue_pattern = gradient_image(ny, nx, 0, 128, 255)
    rgb_image = numpy.zeros((ny, nx, 3))
    rgb_image[:, :, 0] = red_pattern
    rgb_image[:, :, 1] = green_pattern
    rgb_image[:, :, 2] = blue_pattern
    rgb_image = numpy.asarray(rgb_image,dtype=int)
    return rgb_image


def four_corner_gradient(ny, nx):
    red_pattern = gradient_image(ny, nx, ul_val=255, mid_val=128, br_val=0)
    green_pattern = gradient_image(nx, ny, ul_val=255, mid_val=128, br_val=0)
    green_pattern = green_pattern.transpose()
    blue_pattern = gradient_image(ny, nx, 0, 128, 255)
    rgb_image = numpy.zeros((ny, nx, 3))
    rgb_image[:, :, 0] = red_pattern
    rgb_image[:, :, 1] = green_pattern
    rgb_image[:, :, 2] = blue_pattern
    rgb_image = numpy.asarray(rgb_image, dtype=int)
    return rgb_image