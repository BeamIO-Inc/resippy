from __future__ import division

import unittest
import resippy.utils.units.unit_constants as units
from resippy.utils.units import ureg


class TestDefinedUnits(unittest.TestCase):

    def test_microns(self):
        one_micron = 1 * units.micron
        one_meter = 1 * units.meter
        one_meter_plus_1_micron = one_meter + one_micron
        difference = one_meter_plus_1_micron - one_meter
        diff_in_meters = difference.to(units.meter)
        mag = diff_in_meters.magnitude
        assert mag == 1 + 1e-6 - 1
        print("microns unit test passed")

    def test_flicks(self):
        one_flick = 1 * units.flick
        two_flicks = one_flick + 1 * ureg.watt / ureg.cm**2 / ureg.steradian / units.micron
        two_and_one_micro_flicks = two_flicks + 1 * ureg.microflick
        difference = two_and_one_micro_flicks - 1 * ureg.microflick
        assert difference.magnitude == 2

        # test definition on wikipedia:
        one_microflick = 1 * ureg.microflicks
        in_kw_per_steradian_per_cubic_meter = one_microflick.to(
            ureg.kilowatt / ureg.steradian / ureg.meter ** 3)
        assert in_kw_per_steradian_per_cubic_meter.magnitude == 10
        print("microflicks unit test passed")


if __name__ == '__main__':
    unittest.main()
