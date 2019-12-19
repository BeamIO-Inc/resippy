from resippy.test_runner import demo_data_base_dir, demo_data_save_dir

from resippy.utils import file_utils as file_utils
from resippy.utils import envi_utils
from resippy.image_objects.image_factory import ImageFactory
from resippy.spectral.spectrum_factories.spectrum_factory import SpectrumFactory
import resippy.spectral.spectral_image_processing_2d as sip_2d
import resippy.utils.image_utils.image_utils as image_utils
import numpy as np
import imageio
import seaborn
import os

import matplotlib.pyplot as plt


reflectance_cube_fname = file_utils.get_path_from_subdirs(demo_data_base_dir, ["image_data",
                                                                            "hyperspectral",
                                                                            "cooke_city",
                                                                            "self_test",
                                                                            "HyMap",
                                                                            "self_test_refl.img"])

reflectance_envi_image = ImageFactory.envi.from_gdal_file_and_point_calcs(reflectance_cube_fname)
reflectance_image_data = reflectance_envi_image.read_all_image_data_from_disk()
reflectance_image_data = reflectance_image_data.astype(float) * 0.01
ny, nx, nbands = reflectance_image_data.shape

seaborn_color_palette = seaborn.color_palette("RdBu_r")

test_spectrum_fname = file_utils.get_path_from_subdirs(demo_data_base_dir, ["image_data",
                                                                            "hyperspectral",
                                                                            "cooke_city",
                                                                            "self_test",
                                                                            "SPL",
                                                                            "F1",
                                                                            "F1_l.txt"])

spectrum = SpectrumFactory.envi.ascii.EnviAsciiPlotFileFactory.from_ascii_file(test_spectrum_fname)

save_dir = os.path.join(demo_data_save_dir, "spectral_target_detection_demos")
file_utils.make_dir_if_not_exists(save_dir)

def save_rgb():
    print("")
    print("SAVE RGB DEMO")

    red_band = 16
    green_band = 6
    blue_band = 1
    save_image_fname = os.path.join(save_dir, "cooke_city_rgb.png")
    rgb_image = np.zeros((ny, nx, 3))
    rgb_image[:, :, 0] = reflectance_image_data[:, :, red_band]
    rgb_image[:, :, 1] = reflectance_image_data[:, :, green_band]
    rgb_image[:, :, 2] = reflectance_image_data[:, :, blue_band]
    imageio.imwrite(save_image_fname, rgb_image)


def run_ace():
    print("")
    print("RUN ACE DEMO")

    library_spectrum = spectrum.get_spectral_data()

    ace_result = sip_2d.ace(reflectance_image_data, library_spectrum)
    colormapped_ace_result = image_utils.apply_colormap_to_grayscale_image(ace_result,
                                                                           color_palette= seaborn_color_palette)
    save_image_fname = os.path.join(save_dir, "cooke_city_ace.png")
    imageio.imwrite(save_image_fname, colormapped_ace_result)
    print("saved envi cube to: " + save_image_fname)


def run_rx():
    print("")
    print("RUN RX DEMO")

    rx_result = sip_2d.rx_anomaly_detector(reflectance_image_data)
    colormapped_ace_result = image_utils.apply_colormap_to_grayscale_image(rx_result,
                                                                           color_palette= seaborn_color_palette)
    save_image_fname = os.path.join(save_dir, "cooke_city_rx.png")
    imageio.imwrite(save_image_fname, colormapped_ace_result)
    print("saved envi cube to: " + save_image_fname)

def run_pca():
    radiance_cube_fname = file_utils.get_path_from_subdirs(demo_data_base_dir, ["image_data",
                                                                                   "hyperspectral",
                                                                                   "cooke_city",
                                                                                   "self_test",
                                                                                   "HyMap",
                                                                                   "self_test_rad.img"])

    radiance_envi_image = ImageFactory.envi.from_gdal_file_and_point_calcs(radiance_cube_fname)
    radiance_image_data = radiance_envi_image.read_all_image_data_from_disk()

    print("")
    print("RUN PCA DEMO")

    n_pca_bands = 10
    pca_result = sip_2d.compute_image_cube_pca(radiance_image_data, n_components=n_pca_bands, whiten=True)

    for band_number in range(n_pca_bands):
        result = pca_result[:, :, band_number]
        save_image_fname = os.path.join(save_dir, "cooke_city_pca_band_" + str(band_number).zfill(5) + ".png")
        imageio.imwrite(save_image_fname, result)
        print("saved envi cube to: " + save_image_fname)


def main():
    save_rgb()
    run_ace()
    run_rx()
    run_pca()


if __name__ == '__main__':
    main()
