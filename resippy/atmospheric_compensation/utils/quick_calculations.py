from resippy.utils.units import ureg
import numpy


def arm_extent_at_zero_elevation(cloud_deck_height, units):
    earth_radius = 6.371e6 * ureg.parse_units('meters')
    h_clouds = cloud_deck_height * ureg.parse_units(units)

    r = earth_radius.to('meters').magnitude
    r_plus_alt = (earth_radius + h_clouds)
    r_plus_alt = r_plus_alt.to('meters').magnitude

    theta = numpy.arccos(r/r_plus_alt)
    d = numpy.tan(theta) * earth_radius
    return d*2
