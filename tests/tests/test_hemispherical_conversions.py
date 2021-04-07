from __future__ import division

import inspect
import unittest
import numpy
from resippy.atmospheric_compensation.utils import hemisphere_coordinate_conversions


class TestHemisphericalConverstions(unittest.TestCase):

    def test_az_el_to_uv_and_back(self):
        azimuth_angles = numpy.linspace(0, 359, 20)
        elevation_angles = numpy.zeros_like(azimuth_angles) + 10

        azimuths = numpy.deg2rad(azimuth_angles)
        elevations = numpy.deg2rad(elevation_angles)

        uv_x, uv_y = hemisphere_coordinate_conversions.az_el_to_uv_coords(azimuths, elevations)
        az, el = hemisphere_coordinate_conversions.uv_coords_to_az_el_coords(uv_x, uv_y)
        assert numpy.isclose(az, azimuths).all()
        assert numpy.isclose(el, elevations).all()

        print("az el to uv coordinates and back match within a close tolerance.")
        def_name = inspect.stack()[0][3]
        print(def_name + " passed.")

    def test_pixel_yx_to_uv_and_back(self):
        n_uv_pixels = 2048
        y_arr, x_arr = numpy.mgrid[0:n_uv_pixels, 0:n_uv_pixels]
        u, v = hemisphere_coordinate_conversions.uv_pixel_yx_coords_to_uv_coords(n_uv_pixels,
                                                                                 y_arr,
                                                                                 x_arr)
        new_y, new_x = hemisphere_coordinate_conversions.uv_coords_to_uv_pixel_yx_coords(n_uv_pixels, u, v)
        assert numpy.isclose(y_arr, new_y).all()
        assert numpy.isclose(x_arr, new_x).all()
        print("x and y array coordinate to uv coordinates and back match within a close tolerance")
        def_name = inspect.stack()[0][3]
        print(def_name + " passed.")

    def test_uv_to_pixel_yx_boundaries(self):
        n_uv_pixels = 2048
        u = numpy.linspace(0, 1, 10)
        v = numpy.linspace(0, 1, 10)
        y, x = hemisphere_coordinate_conversions.uv_coords_to_uv_pixel_yx_coords(n_uv_pixels, u, v)
        assert x[0] == -0.5
        assert y[-1] == -0.5
        assert x[-1] == n_uv_pixels - 0.5
        assert y[0] == n_uv_pixels - 0.5
        print("uv mappings to pixel coordinates make sense at the boundries")
        def_name = inspect.stack()[0][3]
        print(def_name + " passed.")


if __name__ == '__main__':
    unittest.main()
