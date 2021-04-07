from __future__ import division

import unittest
import numpy
from resippy.utils import coordinate_conversions


class TestCoordinateConversions(unittest.TestCase):

    def test_az_el_to_xyz_and_back(self):
        azimuths = numpy.linspace(0, 360, 100)
        azimuths = numpy.deg2rad(azimuths)
        elevations = numpy.deg2rad(numpy.zeros_like(azimuths) + 10)
        x, y, z = coordinate_conversions.az_el_r_to_xyz(azimuths, elevations, 1)
        az, el, r = coordinate_conversions.xyz_to_az_el_radius(x, y, z)
        assert numpy.isclose(az, azimuths).all()
        assert numpy.isclose(el, elevations).all()
        assert numpy.isclose(r, 1).all()

    def test_xyz_to_az_el_and_back(self):
        y_arr, x_arr = numpy.mgrid[-100:100, -100:100]
        z_input = 1000
        az, el, r = coordinate_conversions.xyz_to_az_el_radius(x_arr, y_arr, z_input)
        x, y, z = coordinate_conversions.az_el_r_to_xyz(az, el, r)
        assert numpy.isclose(z_input, z).all()
        assert numpy.isclose(y_arr, y).all()
        assert numpy.isclose(x_arr, x).all()

    def test_az_el_to_xy_plane_and_back(self):
        azimuths = numpy.linspace(0, 360, 100)
        azimuths = numpy.deg2rad(azimuths)
        elevations = numpy.deg2rad(numpy.zeros_like(azimuths) + 10)
        x, y, z = coordinate_conversions.az_el_to_xy_plane(azimuths, elevations, 100)
        az, el, r = coordinate_conversions.xyz_to_az_el_radius(x, y, z)
        assert numpy.isclose(azimuths, az).all()
        assert numpy.isclose(elevations, el).all()


if __name__ == '__main__':
    unittest.main()
