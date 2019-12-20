import unittest
from resippy.spectral.spectrum_factories.spectrum_factory import SpectrumFactory
import os
import glob
from resippy.utils import file_utils
from resippy.test_runner import demo_data_base_dir


class TestAvirisReader(unittest.TestCase):
    def setUp(self):
        subdirs = ["spectral_libraries", "usgs_splib07", "ASCIIdata"]
        self.usgs_dir = file_utils.get_path_from_subdirs(demo_data_base_dir, subdirs)
        self.test_spectrum_fname_part = "Cordierite-QtzMus_HS346.1B_ASDFRc_AREF.txt"
        self.aviris_ascii_subdirs = ["ASCIIdata_splib07b_cvAVIRISc1995",
                                     "ASCIIdata_splib07b_cvAVIRISc1996",
                                     "ASCIIdata_splib07b_cvAVIRISc1997",
                                     "ASCIIdata_splib07b_cvAVIRISc1998",
                                     "ASCIIdata_splib07b_cvAVIRISc1999",
                                     "ASCIIdata_splib07b_cvAVIRISc2000",
                                     "ASCIIdata_splib07b_cvAVIRISc2001",
                                     "ASCIIdata_splib07b_cvAVIRISc2005",
                                     "ASCIIdata_splib07b_cvAVIRISc2006",
                                     "ASCIIdata_splib07b_cvAVIRISc2009",
                                     "ASCIIdata_splib07b_cvAVIRISc2010",
                                     "ASCIIdata_splib07b_cvAVIRISc2011",
                                     "ASCIIdata_splib07b_cvAVIRISc2012",
                                     "ASCIIdata_splib07b_cvAVIRISc2013",
                                     "ASCIIdata_splib07b_cvAVIRISc2014"
                                     ]
        self.test_spectrum_dir = "ChapterS_SoilsAndMixtures"

    def test_aviris(self):
        print("")
        print("STARTING SPECTRUM FACTORY TEST: AVIRIS")
        n_channels = 224
        for subdir in self.aviris_ascii_subdirs:
            spectrum_factory = SpectrumFactory.usgs.ascii.UsgsAsciiSpectralFactory()
            top_level_dir = os.path.join(self.usgs_dir, subdir)
            spectrum_dir = os.path.join(top_level_dir, self.test_spectrum_dir)
            fullpath = glob.glob(spectrum_dir + "/*" + self.test_spectrum_fname_part + "*")[0]
            fwhm_fname = spectrum_factory.get_fwhm_fname(top_level_dir)
            wavelengths_fname = spectrum_factory.get_wavelengths_fname(top_level_dir)
            assert fwhm_fname is not None
            assert wavelengths_fname is not None

            spectrum = spectrum_factory.from_ascii_file(fullpath)
            assert len(spectrum.get_fwhm()) == n_channels
            assert len(spectrum.get_wavelengths()) == n_channels

            # check to ensure the wavelengths are in microns, max wavelength should around 2.5
            assert(max(spectrum.get_wavelengths()) < 2.6)
            assert(max(spectrum.get_wavelengths()) > 2.4)

            print("Aviris wavelengths are in microns and in the right range, for: " + str(fullpath))

            spectral_data = spectrum.get_spectral_data()

            assert len(spectral_data) == n_channels
            assert spectrum.get_fname() is not None
        print("COMPLETED USGS AVIRIS SPECTRAL UNITS TESTS")


class TestASDReader(unittest.TestCase):
    def setUp(self):
        subdirs = ["spectral_libraries", "usgs_splib07", "ASCIIdata"]
        self.usgs_dir = file_utils.get_path_from_subdirs(demo_data_base_dir, subdirs)
        self.test_spectrum_fname_part = "Cordierite-QtzMus_HS346.1B_ASDFRc_AREF.txt"
        self.asd_ascii_subdirs = ["ASCIIdata_splib07b_cvASD"]
        self.test_spectrum_dir = "ChapterS_SoilsAndMixtures"

    def test_aviris(self):
        print("")
        print("SPECTRUM FACTORY TEST: ASD")
        n_channels = 2151
        for subdir in self.asd_ascii_subdirs:
            aviris_factory = SpectrumFactory.usgs.ascii.UsgsAsciiSpectralFactory()
            top_level_dir = os.path.join(self.usgs_dir, subdir)
            spectrum_dir = os.path.join(top_level_dir, self.test_spectrum_dir)
            fullpath = glob.glob(spectrum_dir + "/*" + self.test_spectrum_fname_part + "*")[0]
            fwhm_fname = aviris_factory.get_fwhm_fname(top_level_dir)
            wavelengths_fname = aviris_factory.get_wavelengths_fname(top_level_dir)
            assert fwhm_fname is not None
            assert wavelengths_fname is not None

            spectrum = aviris_factory.from_ascii_file(fullpath)
            assert len(spectrum.get_fwhm()) == n_channels
            assert len(spectrum.get_wavelengths()) == n_channels

            # check to ensure the wavelengths are in microns, max wavelength should around 2.5
            assert(max(spectrum.get_wavelengths()) < 2.6)
            assert(max(spectrum.get_wavelengths()) > 2.4)

            print("Aviris wavelengths are in microns and in the right range for file: " + str(fullpath))

            spectral_data = spectrum.get_spectral_data()

            assert len(spectral_data) == n_channels
            assert spectrum.get_fname() is not None
        print("ASD SPECTRUM FACTORY TEST PASSED.")


if __name__ == '__main__':
    unittest.main()
