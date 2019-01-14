from __future__ import division

from . import usgs as usgs_spectrum_factories
from . import envi as envi_spectrum_factories


class SpectrumFactory:
    usgs = usgs_spectrum_factories
    envi = envi_spectrum_factories
