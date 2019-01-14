from __future__ import division

from numpy import ndarray
from resippy.utils import spectral_utils as spectral_tools
from resippy.spectral import spectral_image_processing_1d
import resippy.utils.image_utils.image_utils as image_utils


def compute_image_cube_spectral_mean(spectral_image,    # type: ndarray
                                     image_mask=None    # type: ndarray
                                     ):                 # type: (...) -> ndarray
    spectral_image = spectral_tools.flatten_image_cube(spectral_image)
    if image_mask is not None:
        image_mask = image_utils.flatten_image_band(image_mask)
    masked_mean = spectral_image_processing_1d.compute_image_cube_spectral_mean(spectral_image, image_mask)
    return masked_mean


def demean_image_data(spectral_image,       # type: ndarray
                      image_mask=None       # type: ndarray
                      ):                    # type: (...) -> ndarray
    nx, ny, nbands = spectral_tools.get_2d_cube_nx_ny_nbands(spectral_image)
    spectral_image = spectral_tools.flatten_image_cube(spectral_image)
    if image_mask is not None:
        image_mask = image_utils.flatten_image_band(image_mask)
    demeaned_image_data = spectral_image_processing_1d.demean_image_data(spectral_image, image_mask)
    demeaned_image_data = spectral_tools.unflatten_image_cube(demeaned_image_data, nx, ny)
    return demeaned_image_data


def compute_image_cube_spectral_covariance(spectral_image,      # type: ndarray
                                           image_mask=None      # type: ndarray
                                           ):                   # type: (...) -> ndarray
    spectral_image = spectral_tools.flatten_image_cube(spectral_image)
    if image_mask is not None:
        image_mask = image_utils.flatten_image_band(image_mask)
    masked_covariance = spectral_image_processing_1d.compute_image_cube_spectral_covariance(spectral_image, image_mask)
    return masked_covariance


def rx_anomaly_detector(spectral_image,             # type: ndarray
                        spectral_mean=None,         # type: ndarray
                        inverse_covariance=None     # type: ndarray
                        ):                          # type: (...) -> ndarray
    nx, ny, nbands = spectral_tools.get_2d_cube_nx_ny_nbands(spectral_image)
    spectral_image = spectral_tools.flatten_image_cube(spectral_image)
    rx_result = spectral_image_processing_1d.rx_anomaly_detector(spectral_image,
                                                                 spectral_mean,
                                                                 inverse_covariance)
    rx_result = image_utils.unflatten_image_band(rx_result, nx, ny)
    return rx_result


def ace(spectral_image,             # type: ndarray
        target_spectra,             # type: ndarray
        spectral_mean=None,         # type: ndarray
        inverse_covariance=None,    # type: ndarray
        image_mask=None,            # type: ndarray
        ):                          # type: (...) -> ndarray
    nx, ny, nbands = spectral_tools.get_2d_cube_nx_ny_nbands(spectral_image)
    spectral_image = spectral_tools.flatten_image_cube(spectral_image)
    if image_mask is not None:
        image_mask = image_utils.flatten_image_band(image_mask)
    ace_result = spectral_image_processing_1d.ace(spectral_image,
                                                  target_spectra,
                                                  spectral_mean,
                                                  inverse_covariance,
                                                  image_mask)
    ace_result = image_utils.unflatten_image_band(ace_result, nx, ny)
    return ace_result


def ace_demeaned(demeaned_image,            # type: ndarray
                 demeaned_target_spectrum,  # type: ndarray
                 inverse_covariance,        # type: ndarray
                 ):                         # type: (...) -> ndarray
    nx, ny, nbands = spectral_tools.get_2d_cube_nx_ny_nbands(demeaned_image)
    spectral_image = spectral_tools.flatten_image_cube(demeaned_image)
    ace_result = spectral_image_processing_1d.ace_demeaned(spectral_image, demeaned_target_spectrum, inverse_covariance)
    ace_result = image_utils.unflatten_image_band(ace_result, nx, ny)
    return ace_result


def sam(spectral_image,     # type: ndarray
        target_spectra      # type: ndarray
        ):                  # type: (...) -> ndarray
    nx, ny, nbands = spectral_tools.get_2d_cube_nx_ny_nbands(spectral_image)
    spectral_image = spectral_tools.flatten_image_cube(spectral_image)
    sam_result = spectral_image_processing_1d.sam(spectral_image, target_spectra)
    sam_result = image_utils.unflatten_image_band(sam_result, nx, ny)
    return sam_result


def covariance_equalization_mean_centered(input_scene_demeaned,     # type: ndarray
                                          scene_to_match_demeaned,  # type: ndarray
                                          input_scene_mask=None,    # type: ndarray
                                          scene_to_match_mask=None  # type: ndarray
                                          ):                        # type: (...) -> ndarray
    nx, ny, nbands = spectral_tools.get_2d_cube_nx_ny_nbands(input_scene_demeaned)
    flattened_1 = spectral_tools.flatten_image_cube(input_scene_demeaned)
    flattened_2 = spectral_tools.flatten_image_cube(scene_to_match_demeaned)
    if input_scene_mask is not None:
        input_scene_mask = image_utils.flatten_image_band(input_scene_mask)
    if scene_to_match_mask is not None:
        scene_to_match_mask = image_utils.flatten_image_band(scene_to_match_mask)
    scene_1_cov_equalized = spectral_image_processing_1d.covariance_equalization_mean_centered(flattened_1,
                                                                                               flattened_2,
                                                                                               input_scene_mask,
                                                                                               scene_to_match_mask)
    scene_1_cov_equalized = spectral_tools.unflatten_image_cube(scene_1_cov_equalized, nx, ny)
    return scene_1_cov_equalized


# as described in:
# Signature evolution with covariance equalization in oblique hyperspectral imagery
# Proceedings of SPIE - The International Society for Optical Engineering 6233 - May 2006
# Robert A. Leathers, Alan P. Schaum, Trijntje Downes
# Equation 15
def compute_cov_eq_linear_transform_matrix(input_scene,                 # type: ndarray
                                           scene_to_match,              # type: ndarray
                                           input_scene_mask=None,       # type: ndarray
                                           scene_to_match_mask=None     # type: ndarray
                                           ):                           # type: (...) -> ndarray
    flattened_1 = spectral_tools.flatten_image_cube(input_scene)
    flattened_2 = spectral_tools.flatten_image_cube(scene_to_match)
    flattened_mask_1 = image_utils.flatten_image_band(input_scene_mask)
    flattened_mask_2 = image_utils.flatten_image_band(scene_to_match_mask)
    big_l = spectral_image_processing_1d.compute_cov_eq_linear_transform_matrix(flattened_1,
                                                                                flattened_2,
                                                                                flattened_mask_1,
                                                                                flattened_mask_2)
    return big_l
