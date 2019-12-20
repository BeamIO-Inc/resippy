from __future__ import division

import unittest
from resippy.utils import geom_utils


class TestGeometry(unittest.TestCase):

    def test_bounds(self):
        polygon = geom_utils.bounds_2_shapely_polygon(0, 100, 100, 0)
        assert polygon.bounds == (0, 0, 100, 100)
        print("created shapely polygon, bounds are correct.")
        print("geom_utils bounds test passed.")


if __name__ == '__main__':
    unittest.main()
