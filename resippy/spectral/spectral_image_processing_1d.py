from __future__ import division

from numpy import ndarray
import numpy as np
import scipy


def compute_image_cube_spectral_mean(spectral_image,  # type: ndarray
                                     image_mask=None  # type: ndarray
                                     ):               # type: (...) -> ndarray
    if image_mask is None:
        return np.mean(spectral_image, axis=0)
    else:
        mask = np.zeros(spectral_image.shape)
        for i in range(spectral_image.shape[1]):
            mask[:, i] = image_mask
        spectral_image = np.ma.masked_array(spectral_image, mask)
        masked_mean = np.ma.mean(spectral_image, axis=0)
        return masked_mean


def demean_image_data(spectral_image,       # type: ndarray
                      image_mask=None       # type: ndarray
                      ):                    # type: (...) -> ndarray
    image_mean = compute_image_cube_spectral_mean(spectral_image, image_mask)
    return spectral_image - image_mean


def compute_image_cube_spectral_covariance(spectral_image,  # type: ndarray
                                           image_mask=None  # type: ndarray
                                           ):      # type: (...) -> ndarray
    if image_mask is None:
        return np.cov(np.transpose(spectral_image))
    else:
        mask = np.zeros(spectral_image.shape)
        for i in range(spectral_image.shape[1]):
            mask[:, i] = image_mask
        spectral_image = np.ma.masked_array(spectral_image, mask)
        masked_covar = np.ma.cov(np.transpose(spectral_image))
        return masked_covar


# as described in:
# http://www.harrisgeospatial.com/docs/RXAnomalyDetection.html
def rx_anomaly_detector(spectral_image,             # type: ndarray
                        spectral_mean=None,         # type: ndarray
                        inverse_covariance=None     # type: ndarray
                        ):                          # type: (...) -> ndarray
    if spectral_mean is None:
        spectral_mean = compute_image_cube_spectral_mean(spectral_image)
    if inverse_covariance is None:
        scene_covariance = compute_image_cube_spectral_covariance(
            spectral_image)
        inverse_covariance = np.linalg.inv(scene_covariance)
    demeaned_image = np.subtract(spectral_image, spectral_mean)

    # Python 3.5+
    # rx_rt_side = inverse_covariance @ demeaned_image.transpose()
    # Python 2.7 - 3.4
    rx_rt_side = np.dot(inverse_covariance, demeaned_image.transpose())

    rx_left_side = np.multiply(demeaned_image, rx_rt_side.transpose())
    rx_image = np.sum(rx_left_side, axis=1)
    return np.reshape(rx_image, (1, len(rx_image)))


# as described in
# http://www.cis.rit.edu/people/faculty/kerekes/pdfs/Cisz_Thesis.pdf
# equation 4.7
# TODO: find a better way to cite scientific publications and comment in code
# TODO: maybe we could provide our own repo of scientific /
#       supporting material for contributors to use
# TODO: where they could upload/download papers, and
#       link/cite references for ATK / other code
# This def assumes that it is accepting data that has not
# been de-meaned.  If a spectral mean is not provided
# It will compute the mean and then subtract it.  If no inverse
# covariance is given it will compute it from the input
# spectral_image.
def ace(spectral_image,             # type: ndarray
        target_spectra,             # type: ndarray
        spectral_mean=None,         # type: ndarray
        inverse_covariance=None,    # type: ndarray
        image_mask=None,            # type: ndarray
        ):                          # type: (...) -> ndarray

    if spectral_mean is None:
        spectral_mean = compute_image_cube_spectral_mean(
            spectral_image, image_mask)
    if inverse_covariance is None:
        scene_covariance = compute_image_cube_spectral_covariance(
            spectral_image, image_mask)
        inverse_covariance = np.linalg.inv(scene_covariance)
    demeaned_image = np.subtract(spectral_image, spectral_mean)
    demeaned_target_sig_array = target_spectra - spectral_mean

    # Python 3.5+
    # ace_numerator = inverse_covariance @ demeaned_image.transpose()
    # Python 2.7 - 3.4
    ace_numerator = np.dot(inverse_covariance, demeaned_image.transpose())

    ace_numerator = np.multiply(
        demeaned_target_sig_array, ace_numerator.transpose())
    ace_numerator = np.square(np.sum(ace_numerator, axis=1))

    # Python 3.5+
    # ace_den_left = inverse_covariance @ demeaned_target_sig_array.transpose()
    # Python 2.7 - 3.4
    ace_den_left = np.dot(
        inverse_covariance, demeaned_target_sig_array.transpose())

    ace_den_left = np.multiply(
        demeaned_target_sig_array, ace_den_left.transpose())
    ace_den_left = np.sum(ace_den_left)

    # Python 3.5+
    # ace_den_right = inverse_covariance @ demeaned_image.transpose()
    # Python 2.7 - 3.4
    ace_den_right = np.dot(inverse_covariance, demeaned_image.transpose())

    ace_den_right = np.multiply(demeaned_image, ace_den_right.transpose())
    ace_den_right = np.sum(ace_den_right, axis=1)

    ace_image = np.divide(
        ace_numerator, np.multiply(ace_den_left, ace_den_right))
    return ace_image


def ace_demeaned(demeaned_image,            # type: ndarray
                 demeaned_target_spectrum,  # type: ndarray
                 inverse_covariance,        # type: ndarray
                 ):                         # type: (...) -> ndarray
    # Python 3.5+
    # ace_numerator = inverse_covariance @ demeaned_image.transpose()
    # Python 2.7 - 3.4
    ace_numerator = np.dot(inverse_covariance, demeaned_image.transpose())

    ace_numerator = np.multiply(
        demeaned_target_spectrum, ace_numerator.transpose())
    ace_numerator = np.square(np.sum(ace_numerator, axis=1))

    # Python 3.5+
    # ace_den_left = inverse_covariance @ demeaned_target_spectrum.transpose()
    # Python 2.7 - 3.4
    ace_den_left = np.dot(
        inverse_covariance, demeaned_target_spectrum.transpose())

    ace_den_left = np.multiply(
        demeaned_target_spectrum, ace_den_left.transpose())
    ace_den_left = np.sum(ace_den_left)

    # Python 3.5+
    # ace_den_right = inverse_covariance @ demeaned_image.transpose()
    # Python 2.7 - 3.4
    ace_den_right = np.dot(inverse_covariance, demeaned_image.transpose())

    ace_den_right = np.multiply(demeaned_image, ace_den_right.transpose())
    ace_den_right = np.sum(ace_den_right, axis=1)

    ace_image = np.divide(
        ace_numerator, np.multiply(ace_den_left, ace_den_right))
    return ace_image


# as described in
# http://www.cis.rit.edu/people/faculty/kerekes/pdfs/Cisz_Thesis.pdf
# equation 4.1
def sam(spectral_image,     # type: ndarray
        target_spectra      # type: ndarray
        ):                  # type: (...) -> ndarray
    # Python 3.5+
    # numerator = target_spectra @ spectral_image.transpose()
    # Python 2.7 - 3.4
    numerator = np.dot(target_spectra, spectral_image.transpose())

    # Python 3.5+
    # den_left = np.sqrt(target_spectra @ target_spectra)
    # Python 2.7 - 3.4
    den_left = np.sqrt(np.dot(target_spectra, target_spectra))

    den_right = np.sqrt(np.sum(
        np.multiply(spectral_image, spectral_image), axis=1))
    sam_image = numerator / (den_left * den_right)
    return sam_image


# as described in:
# Signature evolution with covariance equalization
# in oblique hyperspectral imagery
#
# Proceedings of SPIE - The International Society for
# Optical Engineering 6233 - May 2006
#
# Robert A. Leathers, Alan P. Schaum, Trijntje Downes
# Equation 11
# This equalizes scene_1 (input_scene_demeaned) to have
# the same scene statistics as scene_2 (scene_to_match)
def covariance_equalization_mean_centered(
        input_scene_demeaned,     # type: ndarray
        scene_to_match_demeaned,  # type: ndarray
        input_scene_mask=None,    # type: ndarray
        scene_to_match_mask=None  # type: ndarray
):                                # type: (...) -> ndarray
    big_l = compute_cov_eq_linear_transform_matrix(
        input_scene_demeaned,
        scene_to_match_demeaned,
        input_scene_mask,
        scene_to_match_mask
    )

    # Python 3.5+
    # input_scene_equalized = big_l @ input_scene_demeaned.transpose()
    # Python 2.7 - 3.4
    input_scene_equalized = np.dot(big_l, input_scene_demeaned.transpose())

    return input_scene_equalized.transpose()


# as described in:
# Signature evolution with covariance equalization in
# oblique hyperspectral imagery
#
# Proceedings of SPIE - The International Society for
# Optical Engineering 6233 - May 2006
#
# Robert A. Leathers, Alan P. Schaum, Trijntje Downes
# Equation 15
def compute_cov_eq_linear_transform_matrix(
        input_scene,                 # type: ndarray
        scene_to_match,              # type: ndarray
        input_scene_mask=None,       # type: ndarray
        scene_to_match_mask=None     # type: ndarray
):                                   # type: (...) -> ndarray
    n_bands = np.shape(input_scene)[1]
    cov_1 = compute_image_cube_spectral_covariance(
        input_scene, input_scene_mask)
    cov_2 = compute_image_cube_spectral_covariance(
        scene_to_match, scene_to_match_mask)
    cov_1_to_minus_one_half = scipy.linalg.fractional_matrix_power(cov_1, -0.5)
    cov_2_to_plus_one_half = scipy.linalg.fractional_matrix_power(cov_2, 0.5)
    q = np.eye(n_bands)

    # Python 3.5+
    # big_l = cov_2_to_plus_one_half @ q @ cov_1_to_minus_one_half
    # Python 2.7 - 3.4
    big_l = np.dot(np.dot(cov_2_to_plus_one_half, q), cov_1_to_minus_one_half)

    return big_l
