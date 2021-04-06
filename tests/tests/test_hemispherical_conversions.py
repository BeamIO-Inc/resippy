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


if __name__ == '__main__':
    unittest.main()
