import numpy
from resippy.utils.lens_distortion_utils.distortion_models.brown_conrady import BrownConradyDistortionModel
from resippy.utils.image_utils import image_utils

x_center = 0.0000
y_center = 0.0000
radial_distortion_params = numpy.asarray((0, 0))
tangential_distortion_params = numpy.asarray((0, 0, 0))
distortion_model = BrownConradyDistortionModel(x_center,
                                               y_center,
                                               radial_distortion_params,
                                               tangential_distortion_params)

nx = 640
ny = 481
spacing = 5e-6

image_plane_grid = image_utils.create_image_plane_grid(nx, ny, x_spacing=spacing, y_spacing=spacing)

undistorted_locs = distortion_model.compute_undistorted_image_plane_locations(image_plane_grid[0], image_plane_grid[1])
stop = 1