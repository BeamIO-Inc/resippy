from __future__ import division

import unittest
import numpy
from resippy.atmospheric_compensation.utils import hemisphere_coordinate_conversions
from resippy.utils import coordinate_conversions
from resippy.atmospheric_compensation.arm_climate_model import ArmClimateModel


class TestHemisphereProjections(unittest.TestCase):

    def test_projection_to_cloud_deck_and_back(self):
        n_uv_pixels = 2048
        arm_image = numpy.zeros((n_uv_pixels, n_uv_pixels))
        arm_model = ArmClimateModel.from_numpy_array(arm_image)
        arm_model.center_xyz_location = (0, 0, 0)
        cloud_x, cloud_y, cloud_z = arm_model.project_uv_image_pixels_to_cloud_deck(1000)
        az_1, el_1, r = coordinate_conversions.xyz_to_az_el_radius(cloud_x, cloud_y, cloud_z)
        uv_pixels_y, uv_pixels_x = numpy.mgrid[0:n_uv_pixels, 0:n_uv_pixels]
        az_2, el_2 = hemisphere_coordinate_conversions.uv_pixel_yx_coords_to_az_el(n_uv_pixels, uv_pixels_y, uv_pixels_x)
        stop = 1


if __name__ == '__main__':
    unittest.main()
