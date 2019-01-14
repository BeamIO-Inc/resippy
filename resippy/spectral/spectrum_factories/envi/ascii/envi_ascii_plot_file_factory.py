from __future__ import division

import numpy as np
from resippy.spectral.spectrum import Spectrum
from resippy.utils import string_utils

WAVELENGTH_UNITS = "nanometers"
SPECTRUM_UNITS = "reflectance_0-1"


class EnviAsciiPlotFileFactory:
    @classmethod
    def from_ascii_file(cls, ascii_file     # type: str
                        ):                  # type: (...) -> Spectrum
        with open(ascii_file) as f:
            content = f.readlines()

        wavelengths = []
        spectral_data = []
        wavelengths_and_reflectances = content[3:]
        for text in wavelengths_and_reflectances:
            text = string_utils.remove_newlines(text)
            text = ' '.join(text.split())
            w, r = text.split()
            wavelengths.append(float(w))
            spectral_data.append(float(r))

        spectral_data = np.array(spectral_data)
        wavelengths = np.array(wavelengths)

        spectrum = Spectrum()
        spectrum.set_wavelength_units(WAVELENGTH_UNITS)
        spectrum.set_wavelengths(wavelengths)
        spectrum.set_spectral_data(spectral_data)
        spectrum.set_spectrum_units(SPECTRUM_UNITS)
        return spectrum
