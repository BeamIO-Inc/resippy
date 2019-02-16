from __future__ import division

import numpy as np
import csv
from resippy.spectral.spectrum import Spectrum

WAVELENGTH_UNITS = "nanometers"
SPECTRUM_UNITS = "reflectance_0-1"


class CsvAsciiFileFactory:
    # TODO handle variable unit types?
    @classmethod
    def from_csv_file(cls, csv_file,     # type: str
                      delimiter=',',  # type: str
                      nheader=1,  # type: int
                      wave_index=0,  # type: int
                      reflectance_index=1  # type: int
                      ):                  # type: (...) -> Spectrum

        wavelengths = []
        spectral_data = []
        with open(csv_file) as f:
            reader = csv.reader(f, delimiter=delimiter)

            for i in range(nheader):
                next(reader)

            for row in reader:
                wave = row[wave_index]
                ref = row[reflectance_index]
                wavelengths.append(float(wave))
                spectral_data.append(float(ref))

        spectral_data = np.array(spectral_data)
        wavelengths = np.array(wavelengths)

        spectrum = Spectrum()
        spectrum.set_wavelength_units(WAVELENGTH_UNITS)
        spectrum.set_wavelengths(wavelengths)
        spectrum.set_spectral_data(spectral_data)
        spectrum.set_spectrum_units(SPECTRUM_UNITS)
        return spectrum
