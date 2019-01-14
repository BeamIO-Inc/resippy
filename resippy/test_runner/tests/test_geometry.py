from __future__ import division

import unittest
from resippy.utils import geom_utils


class TestGeometry(unittest.TestCase):

    def test_bounds(self):
        geom_utils.bounds_2_shapely_polygon(0, 100, 100, 0)


if __name__ == '__main__':
    unittest.main()
