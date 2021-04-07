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
        assert numpy.isclose(az, azimuths)
        assert numpy.isclose(el, elevations)
        assert numpy.isclose(r, 1)


if __name__ == '__main__':
    unittest.main()
