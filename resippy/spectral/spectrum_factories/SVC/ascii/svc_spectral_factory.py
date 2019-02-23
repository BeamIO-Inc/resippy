from __future__ import division

import numpy as np
from resippy.spectral.spectrum import Spectrum

WAVELENGTH_UNITS = "nanometers"
SPECTRUM_UNITS = "reflectance_0-1"


class SvcAsciiFileFactory:
    @classmethod
    def from_sig_file(cls, ascii_file     # type: str
                          ):                  # type: (...) -> Spectrum
        with open(ascii_file) as f:
            content = f.readlines()

        data_ind = 0
        for ind in range(len(content)):
            data_ind += 1
            if content[ind].startswith('data'):
                break

        wavelengths = []
        spectral_data = []
        while data_ind < len(content):
            wave, _, _, ref = content[data_ind].split()

            data_ind += 1
            wavelengths.append(float(wave))
            spectral_data.append(float(ref)/100.0)

        spectral_data = np.array(spectral_data)
        wavelengths = np.array(wavelengths)

        spectrum = Spectrum()
        spectrum.set_wavelength_units(WAVELENGTH_UNITS)
        spectrum.set_wavelengths(wavelengths)
        spectrum.set_spectral_data(spectral_data)
        spectrum.set_spectrum_units(SPECTRUM_UNITS)
        return spectrum
