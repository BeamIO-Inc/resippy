from __future__ import division

from numpy import ndarray
import numpy as np
import copy


class Spectrum:
    def __init__(self):
        self._wavelengths = None
        self._spectral_data = None
        self._fwhm = None
        self._spectrum_units = None
        self._wavelength_units = None
        self._bbl = None
        self._nodata_val = None
        self._fname = None

    def get_fname(self):            # type: (...) -> str
        return self._fname

    def set_fname(self,
                  fname             # type: str
                  ):                # type: (...) -> None
        self._fname = fname

    def get_nodata_val(self):       # type: (...) -> float
        return self._nodata_val

    def set_nodata_val(self,
                       nodata_val   # type: float
                       ):           # type: (...) -> None
        self._nodata_val = nodata_val

    def get_spectral_data(self,
                          nodata_to_nan=True,   # type: bool
                          with_units=False      # type: bool
                          ):                    # type: (...) -> ndarray
        spectral_data = copy.deepcopy(self._spectral_data)
        if nodata_to_nan:
            # spectral_data_nodata_replaced = copy.deepcopy(self._spectral_data)
            spectral_data[np.where(spectral_data == self.get_nodata_val())] = np.nan
        if with_units:
            spectral_data = spectral_data * self.get_spectrum_units()
        return spectral_data

    def set_spectral_data(self,
                          spectral_data     # type: ndarray
                          ):                # type: (...) -> None
        self._spectral_data = spectral_data

    def get_wavelengths(self,
                        with_units=False    # type: bool
                        ):                  # type: (...) -> ndarray
        wavelengths = self._wavelengths
        if with_units:
            wavelengths = wavelengths * self.get_wavelength_units()
        return wavelengths

    def set_wavelengths(self,
                        wavelengths     # type: ndarray
                        ):              # type: (...) -> None
        self._wavelengths = wavelengths

    def get_fwhm(self,
                 with_units=False   # type: bool
                 ):                 # type: (...) -> ndarray
        fwhm = self._fwhm
        if with_units:
            fwhm = fwhm * self.get_wavelength_units()
        return fwhm

    def set_fwhm(self,
                 fwhm       # type: ndarray
                 ):         # type: (...) -> None
        self._fwhm = fwhm

    def get_spectrum_units(self):       # type: (...) -> ndarray
        return self._spectrum_units

    def set_spectrum_units(self,
                           units        # type: str
                           ):           # type: (...) -> None
        self._spectrum_units = units

    def get_wavelength_units(self):     # type: (...) -> str
        return self._wavelength_units

    def set_wavelength_units(self,
                             units      # type: str
                             ):         # type: (...) -> None
        self._wavelength_units = units

    # TODO convert wavelength units if needed
    def resample_spectrum_direct(self,
                                 rsr_spectrums,     # type: List[Spectrum]
                                 rsr_band_centers   # type: ndarray
                                 ):                 # type: (...) -> Spectrum

        spectra = []
        for rsr_spectrum in rsr_spectrums:
            rsr_wave = rsr_spectrum.get_wavelengths()
            rsr_data = rsr_spectrum.get_spectral_data()

            rsr_data_resamp = np.interp(self.get_wavelengths(), rsr_wave, rsr_data, left=0, right=0)
            normed = self.get_spectral_data() * rsr_data_resamp

            # import matplotlib.pyplot as plt
            # # plt.plot(self.get_wavelengths(), rsr_data_resamp)
            #
            #
            # plt.plot(self.get_wavelengths(), self.get_spectral_data())
            # plt.plot(self.get_wavelengths(), normed)

            b_center = np.trapz(normed, self.get_wavelengths())
            b_width = np.trapz(rsr_data_resamp, self.get_wavelengths())

            spectra.append(b_center/b_width)

        new_spec = Spectrum()
        new_spec.set_wavelengths(rsr_band_centers)
        new_spec.set_spectral_data(np.array(spectra))
        new_spec.set_spectrum_units(self.get_spectrum_units())

        return new_spec
