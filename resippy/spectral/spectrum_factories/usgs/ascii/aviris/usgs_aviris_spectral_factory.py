from __future__ import division

from numpy import ndarray
from resippy.spectral.spectrum import Spectrum
import os
import glob
import numpy as np
from resippy.utils.units import ureg
from resippy.utils.units import unit_constants

NODATA_VAL = -1.23e34
WAVELENGTH_UNITS = ureg.microns
SPECTRUM_UNITS = unit_constants.reflectance_zero_to_one


class UsgsAsciiAvirisSpectralFactory:
    @classmethod
    def from_ascii_file(cls, filename   # type: str
                        ):              # type: (...) -> Spectrum
        metadata_dir = os.path.dirname(os.path.dirname(filename))
        wavelengths_fname = cls.get_wavelengths_fname(metadata_dir)
        wavelengths = cls.read_ascii_wavelengths(wavelengths_fname)
        fwhm_fname = cls.get_fwhm_fname(metadata_dir)
        fwhm = cls.read_ascii_fwhm(fwhm_fname)
        with open(filename) as f:
            content = f.readlines()
        content = [x.strip() for x in content][1:]
        reflectance = np.array([float(x) for x in content])
        spectrum = Spectrum()
        spectrum.set_wavelength_units(WAVELENGTH_UNITS)
        spectrum.set_wavelengths(wavelengths)
        spectrum.set_fwhm(fwhm)
        spectrum.set_spectral_data(reflectance)
        spectrum.set_spectrum_units(SPECTRUM_UNITS)
        spectrum.set_nodata_val(NODATA_VAL)
        spectrum.set_fname(filename)
        return spectrum

    @staticmethod
    def get_fwhm_fname(base_dir     # type: str
                       ):           # type: (...) -> str
        txt_files = glob.glob(base_dir + "/*.txt")
        for txt_file in txt_files:
            basename_lower = str.lower(os.path.basename(txt_file))
            if "resolution" in basename_lower or \
                    "bandpass" in basename_lower:
                return txt_file

    @staticmethod
    def get_wavelengths_fname(base_dir     # type: str
                              ):           # type: (...) -> str
        txt_files = glob.glob(base_dir + "/*.txt")
        for txt_file in txt_files:
            basename_lower = str.lower(os.path.basename(txt_file))
            if "wavelength" in basename_lower or \
                    "waves" in basename_lower:
                return txt_file

    @staticmethod
    def read_ascii_wavelengths(filename     # type: str
                               ):           # type: (...) -> ndarray
        with open(filename) as f:
            content = f.readlines()
        content = [x.strip() for x in content][1:]
        wavelengths = np.array([float(x) for x in content])
        return wavelengths

    @staticmethod
    def read_ascii_fwhm(filename     # type: str
                        ):           # type: (...) -> ndarray
        with open(filename) as f:
            content = f.readlines()
        content = [x.strip() for x in content][1:]
        fwhm = np.array([float(x) for x in content])
        return fwhm
