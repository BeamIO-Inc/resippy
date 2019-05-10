from __future__ import division

import unittest
from resippy.utils import spectral_utils
from resippy.spectral import spectral_image_processing_1d as sp1d
from resippy.spectral import spectral_image_processing_2d as sp2d
from resippy.utils.image_utils import image_utils
import numpy as np
import copy

import logging

nx = 300
ny = 200
nbands = 30
x_loc = 40
y_loc = 60


class TestSpectralTools(unittest.TestCase):
    print("SHAPE CONSISTENCY TEST")
    print("")

    def test_spectraltool_imageutils_shape_consistency(self):
        logging.debug("SPECTRAL TOOLS AND IMAGE UTILS SHAPE CONSISTENCY TEST")
        image_cube = image_utils.create_uniform_image_data(
            nx, ny, nbands, values=0, dtype=np.float32)
        im_nx, im_ny, im_nbands = spectral_utils.get_2d_cube_nx_ny_nbands(
            image_cube)

        assert nx == im_nx
        assert ny == im_ny
        assert nbands == im_nbands
        logging.debug(
            "dimensionality consistency between image_utils "
            "and spectral_tools passed"
        )
    print("shape consistency test passed")

    def test_flatten_image_cube(self):
        print("")
        print("FLATTEN IMAGE CUBE TEST")
        image_cube = image_utils.create_uniform_image_data(
            nx, ny, nbands, values=0, dtype=np.float32)
        image_cube[y_loc, x_loc, :] = np.arange(0, nbands)
        flattened_image = spectral_utils.flatten_image_cube(image_cube)
        unflattend_image_cube = spectral_utils.unflatten_image_cube(
            flattened_image, nx, ny)

        assert (unflattend_image_cube == image_cube).all()
        print("flattening and unflattening image cube matches with original")

    def test_masked_mean_1d(self):
        print("")
        print("MASKED MEAN TEST 1D")
        image_cube = image_utils.create_uniform_image_data(
            nx, ny, nbands, values=0, dtype=np.float32)
        image_cube[y_loc, x_loc, :] = np.arange(0, nbands)
        flattened_image = spectral_utils.flatten_image_cube(image_cube)

        image_mask = np.zeros((ny, nx))
        image_mask[y_loc, x_loc] = 1
        flattened_mask = image_utils.flatten_image_band(image_mask)

        raw_mean = sp1d.compute_image_cube_spectral_mean(flattened_image)
        masked_mean = sp1d.compute_image_cube_spectral_mean(
            flattened_image, flattened_mask)

        assert len(masked_mean) == nbands
        assert len(raw_mean) == nbands
        logging.debug("length of means equals number of spectral bands")

        assert masked_mean.max() == 0
        assert raw_mean.max() != 0
        logging.debug("1d image cube raw and masked means have different results")
        logging.debug("masked_mean_1d test passed")
        print("MASKED MEAN TEST 1D - TEST PASSED")

    def test_masked_mean_2d(self):
        print("")
        print("MASKED MEAN TEST 2D")
        image_cube = image_utils.create_uniform_image_data(
            nx, ny, nbands, values=0, dtype=np.float32)
        image_cube[y_loc, x_loc, :] = np.arange(0, nbands)

        image_mask = np.zeros((ny, nx))
        image_mask[y_loc, x_loc] = 1

        raw_mean = sp2d.compute_image_cube_spectral_mean(image_cube)
        masked_mean = sp2d.compute_image_cube_spectral_mean(
            image_cube, image_mask)

        assert len(raw_mean) == nbands
        assert len(masked_mean) == nbands
        logging.debug("length of means equals number of spectral bands")

        assert masked_mean.max() == 0
        assert raw_mean.max() != 0
        logging.debug(
            "2d image cube raw and masked means have "
            "different results, test passed"
        )

    def test_masked_covar_1d(self):
        print("")
        print("MASKED COVARIANCE TEST 1D")
        image_cube = image_utils.create_uniform_image_data(
            nx, ny, nbands, values=0, dtype=np.float32)
        image_cube[y_loc, x_loc, :] = np.arange(0, nbands)
        flattened_image = spectral_utils.flatten_image_cube(image_cube)

        image_mask = np.zeros((ny, nx))
        image_mask[y_loc, x_loc] = 1
        flattened_mask = image_utils.flatten_image_band(image_mask)

        raw_covar = sp1d.compute_image_cube_spectral_covariance(
            flattened_image)
        masked_covar = sp1d.compute_image_cube_spectral_covariance(
            flattened_image, flattened_mask)

        assert masked_covar.shape == (nbands, nbands)
        assert raw_covar.shape == (nbands, nbands)
        logging.debug("covariance shape is (nbands x nbands)")

        assert masked_covar.max() == 0
        assert raw_covar.max() != 0
        logging.debug(
            "1d image cube raw and masked covariances have different results"
        )
        print("MASKED_COVAR_1D TEST PASSED")

    def test_masked_covar_2d(self):
        print("")
        print("MASKED COVARIANCE TEST 2D")
        image_cube = image_utils.create_uniform_image_data(
            nx, ny, nbands, values=0, dtype=np.float32)
        image_cube[y_loc, x_loc, :] = np.arange(0, nbands)
        image_mask = np.zeros((ny, nx))
        image_mask[y_loc, x_loc] = 1

        raw_covar = sp2d.compute_image_cube_spectral_covariance(image_cube)
        masked_covar = sp2d.compute_image_cube_spectral_covariance(
            image_cube, image_mask)

        assert masked_covar.shape == (nbands, nbands)
        assert raw_covar.shape == (nbands, nbands)
        logging.debug("covariance shape is (nbands x nbands)")

        assert masked_covar.max() == 0
        assert raw_covar.max() != 0
        logging.debug(
            "2d image cube raw and masked covariances have different results"
        )
        print("MASKED COVAR 2D TEST PASSED")

    def test_rx_1d(self):
        print("")
        print("RX ANAMOLY TEST 1D")
        image_cube = np.random.random((ny, nx, nbands))
        signal_x_axis = np.arange(0, nbands) / nbands * 2 * np.pi
        signal_to_embed = np.sin(signal_x_axis) * 100
        image_cube[y_loc, x_loc, :] = signal_to_embed
        flattened_image = spectral_utils.flatten_image_cube(image_cube)
        image_mask = np.zeros((ny, nx))
        image_mask[y_loc, x_loc] = 1
        flattened_mask = image_utils.flatten_image_band(image_mask)

        masked_mean = sp1d.compute_image_cube_spectral_mean(
            flattened_image, flattened_mask)
        masked_covar = sp1d.compute_image_cube_spectral_covariance(
            flattened_image, flattened_mask)
        masked_inv_cov = np.linalg.inv(masked_covar)

        detection_result = sp1d.rx_anomaly_detector(
            flattened_image, masked_mean, masked_inv_cov)
        detection_result_2d = image_utils.unflatten_image_band(
            detection_result, nx, ny)
        detection_max_locs_y_x = np.where(
            detection_result_2d == detection_result_2d.max())
        detection_max_y = detection_max_locs_y_x[0][0]
        detection_max_x = detection_max_locs_y_x[1][0]

        assert detection_max_y == y_loc
        assert detection_max_x == x_loc
        assert len(detection_result.shape) == 1
        assert detection_result.shape[0] == nx*ny

        logging.debug(
            "location of highest detection return matches x/y "
            "location of embedded signal"
        )
        print("1d rx passed ")
        print("")

    def test_rx_2d(self):
        print("")
        print("RX ANAMOLY TEST 2D")
        image_cube = np.random.random((ny, nx, nbands))
        signal_x_axis = np.arange(0, nbands) / nbands * 2 * np.pi
        signal_to_embed = np.sin(signal_x_axis) * 100
        image_cube[y_loc, x_loc, :] = signal_to_embed
        image_mask = np.zeros((ny, nx))
        image_mask[y_loc, x_loc] = 1

        masked_mean = sp2d.compute_image_cube_spectral_mean(
            image_cube, image_mask)
        masked_covar = sp2d.compute_image_cube_spectral_covariance(
            image_cube, image_mask)
        masked_inv_cov = np.linalg.inv(masked_covar)

        detection_result = sp2d.rx_anomaly_detector(
            image_cube, masked_mean, masked_inv_cov)
        detection_max_locs_y_x = np.where(
            detection_result == detection_result.max())
        detection_max_y = detection_max_locs_y_x[0][0]
        detection_max_x = detection_max_locs_y_x[1][0]

        assert detection_max_y == y_loc
        assert detection_max_x == x_loc
        logging.debug(
            "location of highest detection return matches x/y "
            "location of embedded signal"
        )
        logging.debug("2d rx test passed.")

    def test_ace_1d(self):
        print("")
        print("ACE TEST 1D")
        image_cube = np.random.random((ny, nx, nbands))
        signal_x_axis = np.arange(0, nbands) / nbands * 2 * np.pi
        signal_to_embed = np.sin(signal_x_axis) * 100
        image_cube[y_loc, x_loc, :] = image_cube[
            y_loc, x_loc, :] + signal_to_embed
        flattened_image = spectral_utils.flatten_image_cube(image_cube)
        image_mask = np.zeros((ny, nx))
        image_mask[y_loc, x_loc] = 1
        flattened_mask = image_utils.flatten_image_band(image_mask)

        masked_mean = sp1d.compute_image_cube_spectral_mean(
            flattened_image, flattened_mask)
        masked_covar = sp1d.compute_image_cube_spectral_covariance(
            flattened_image, flattened_mask)

        masked_inv_cov = np.linalg.inv(masked_covar)
        detection_result = sp1d.ace(
            flattened_image, signal_to_embed, masked_mean, masked_inv_cov)

        detection_result_2d = image_utils.unflatten_image_band(
            detection_result, nx, ny)
        detection_max_locs_y_x = np.where(
            detection_result_2d == detection_result_2d.max())
        detection_max_y = detection_max_locs_y_x[0][0]
        detection_max_x = detection_max_locs_y_x[1][0]

        assert detection_max_y == y_loc
        assert detection_max_x == x_loc
        assert len(detection_result.shape) == 1
        assert detection_result.shape[0] == nx*ny
        logging.debug(
            "location of highest detection return matches x/y "
            "location of embedded signal"
        )
        print("1D ACE TEST PASSED")

    def test_ace_2d(self):
        print("")
        print("ACE TEST 2D")
        image_cube = np.random.random((ny, nx, nbands))
        signal_x_axis = np.arange(0, nbands) / nbands * 2 * np.pi
        signal_to_embed = np.sin(signal_x_axis) * 100
        image_cube[y_loc, x_loc, :] = image_cube[
            y_loc, x_loc, :] + signal_to_embed
        image_mask = np.zeros((ny, nx))
        image_mask[y_loc, x_loc] = 1

        masked_mean = sp2d.compute_image_cube_spectral_mean(
            image_cube, image_mask)
        masked_covar = sp2d.compute_image_cube_spectral_covariance(
            image_cube, image_mask)

        masked_inv_cov = np.linalg.inv(masked_covar)
        detection_result = sp2d.ace(
            image_cube, signal_to_embed, masked_mean, masked_inv_cov)

        detection_max_locs_y_x = np.where(
            detection_result == detection_result.max())
        detection_max_y = detection_max_locs_y_x[0][0]
        detection_max_x = detection_max_locs_y_x[1][0]

        assert detection_max_y == y_loc
        assert detection_max_x == x_loc
        logging.debug(
            "location of highest detection return matches x/y "
            "location of embedded signal"
        )
        logging.debug("2d ace test passed.")

    def test_sam_1d(self):
        print("")
        print("SAM TEST 1D")
        random_cube = np.random.random((ny, nx, nbands))
        signal_x_axis = np.arange(0, nbands) / nbands * 2 * np.pi
        signal_to_embed = np.sin(signal_x_axis) * 100
        background_to_embed = np.cos(signal_x_axis) * 100
        image_cube = random_cube + background_to_embed
        image_cube[y_loc, x_loc, :] = image_cube[
            y_loc, x_loc, :] + signal_to_embed
        flattened_image = spectral_utils.flatten_image_cube(image_cube)
        detection_result = sp1d.sam(flattened_image, signal_to_embed)

        detection_result_2d = image_utils.unflatten_image_band(
            detection_result, nx, ny)
        detection_max_locs_y_x = np.where(
            detection_result_2d == detection_result_2d.max())
        detection_max_y = detection_max_locs_y_x[0][0]
        detection_max_x = detection_max_locs_y_x[1][0]

        assert detection_max_y == y_loc
        assert detection_max_x == x_loc
        logging.debug(
            "location of highest detection return matches x/y "
            "location of embedded signal"
        )
        print("1-D SAM TEST PASSED")

    def test_sam_2d(self):
        print("")
        print("SAM TEST 2D")
        random_cube = np.random.random((ny, nx, nbands))
        signal_x_axis = np.arange(0, nbands) / nbands * 2 * np.pi
        signal_to_embed = np.sin(signal_x_axis) * 100
        background_to_embed = np.cos(signal_x_axis) * 100
        image_cube = random_cube + background_to_embed
        image_cube[y_loc, x_loc, :] = image_cube[
            y_loc, x_loc, :] + signal_to_embed
        detection_result = sp2d.sam(
            image_cube, signal_to_embed)

        detection_max_locs_y_x = np.where(
            detection_result == detection_result.max())
        detection_max_y = detection_max_locs_y_x[0][0]
        detection_max_x = detection_max_locs_y_x[1][0]

        assert detection_max_y == y_loc
        assert detection_max_x == x_loc
        logging.debug(
            "location of highest detection return matches x/y "
            "location of embedded signal"
        )
        print("2d sam test passed.")

    # TODO: possibly remove, this might be redundant with test_sam_ace_compare
    def test_sam_ace_sanity_check(self):
        print("")
        print("SAM / ACE COMPARISON AND SANITY CHECK TEST")
        random_cube = np.random.random((ny, nx, nbands))
        signal_x_axis = np.arange(0, nbands) / nbands * 2 * np.pi
        signal_to_embed = np.sin(signal_x_axis) * 100
        background_to_embed = np.cos(signal_x_axis) * 100
        image_cube = random_cube + background_to_embed
        image_cube[y_loc, x_loc, :] = image_cube[
            y_loc, x_loc, :] + signal_to_embed
        flattened_image = spectral_utils.flatten_image_cube(image_cube)

        demeaned_target_sig_array = signal_to_embed

        inverse_covariance = np.eye(nbands)
        signal_x_axis = np.arange(0, nbands) / nbands * 2 * np.pi
        signal_to_embed = np.sin(signal_x_axis) * 100
        target_spectra = signal_to_embed

        # checking ace / sam numerator right hand side

        # Python 3.5+
        # ace_numerator = inverse_covariance @ flattened_image.transpose()
        # Python 2.7 - 3.4
        ace_numerator = np.dot(inverse_covariance, flattened_image.transpose())

        ace_numerator = np.multiply(target_spectra, ace_numerator.transpose())
        ace_numerator = np.square(np.sum(ace_numerator, axis=1))

        # Python 3.5+
        # ace_den_left = (
        #    inverse_covariance @ demeaned_target_sig_array.transpose())
        # Python 2.7 - 3.4
        ace_den_left = np.dot(
            inverse_covariance, demeaned_target_sig_array.transpose())

        ace_den_left = np.multiply(
            demeaned_target_sig_array, ace_den_left.transpose())
        ace_den_left = np.sum(ace_den_left)

        # Python 3.5+
        # ace_den_right = inverse_covariance @ flattened_image.transpose()
        # Python 2.7 - 3.4
        ace_den_right = np.dot(inverse_covariance, flattened_image.transpose())

        ace_den_right = np.multiply(flattened_image, ace_den_right.transpose())
        ace_den_right = np.sum(ace_den_right, axis=1)

        # Python 3.5+
        # sam_numerator = target_spectra @ flattened_image.transpose()
        # Python 2.7 - 3.4
        sam_numerator = np.dot(target_spectra, flattened_image.transpose())

        # Python 3.5+
        # sam_den_left = np.sqrt(target_spectra @ target_spectra)
        # Python 2.7 - 3.4
        sam_den_left = np.sqrt(np.dot(target_spectra, target_spectra))

        sam_den_right = np.sqrt(np.sum(np.multiply(
            flattened_image, flattened_image), axis=1))

        np.testing.assert_allclose(ace_numerator, np.square(sam_numerator))
        np.testing.assert_allclose(ace_den_left, np.square(sam_den_left))
        np.testing.assert_allclose(ace_den_right, np.square(sam_den_right))

        logging.debug(
            "sam and ace portions of numerators and denominators are "
            "close when the covariance is an identify matrix."
        )
        print("SAM / ACE SANITY CHECK TEST PASSED")

    def test_sam_ace_compare(self):
        print("")
        print("SAM / ACE EQUALITY TEST WHEN INVERSE COVARIANCE IS IDENTITY")
        random_cube = np.random.random((ny, nx, nbands))
        signal_x_axis = np.arange(0, nbands) / nbands * 2 * np.pi
        signal_to_embed = np.sin(signal_x_axis) * 100
        background_to_embed = np.cos(signal_x_axis) * 100
        image_cube = random_cube + background_to_embed
        image_cube[y_loc, x_loc, :] = image_cube[
            y_loc, x_loc, :] + signal_to_embed
        flattened_image = spectral_utils.flatten_image_cube(image_cube)

        ace_inv_cov = np.eye(nbands)
        ace_mean = np.zeros(nbands)

        ace_detection_result = sp1d.ace(
            flattened_image, signal_to_embed, ace_mean, ace_inv_cov)
        sam_detection_result = sp1d.sam(flattened_image, signal_to_embed)

        np.testing.assert_allclose(ace_detection_result, np.square(
            sam_detection_result))
        logging.debug(
            "SAM squared is equivalent to ACE if the inverse "
            "covariance is the identify matrix"
        )
        print("ACE / SAM comparison test passed.")

    def test_covariance_equalization(self):
        print("")
        print("COVARIANCE EQUALIZATION TEST")
        scene_1 = np.random.random((ny, nx, nbands))
        scene_1 = spectral_utils.flatten_image_cube(scene_1)
        scene_1 = scene_1 + (np.sin(
            np.arange(0, nbands) / nbands * (2 * np.pi)) / 4 + 0.75)
        scene_2 = scene_1 * 2.0
        scene_1_equalized = sp1d.covariance_equalization_mean_centered(
            scene_1, scene_2)
        scene_2_cov = sp1d.compute_image_cube_spectral_covariance(scene_2)
        scene_1_equalized_cov = sp1d.compute_image_cube_spectral_covariance(
            scene_1_equalized)
        np.testing.assert_almost_equal(scene_2_cov, scene_1_equalized_cov)
        logging.debug(
            "equalized covariance from scene_1 matches covariance of scene_2"
        )
        print("COVARIANCE EQUALIZATION TEST PASSED")

    def test_cov_eq_target_detection(self):
        print("")
        print("COVARIANCE EQUALIZATION TARGET DETECTION TEST")
        scene_1 = (np.random.random((ny, nx, nbands))) * 0.1 + 0.95
        signal_x_axis = np.arange(0, nbands) / nbands * 2 * np.pi
        signal_to_embed = ((np.sin(signal_x_axis) + 1) / 2.0) * 10
        scene_1[y_loc, x_loc, :] = scene_1[y_loc, x_loc, :] * signal_to_embed

        scene_2 = copy.deepcopy(scene_1)
        scene_1 = scene_1 * signal_to_embed

        scene_1_ace_result = sp2d.ace(scene_1, signal_to_embed)
        scene_2_ace_result = sp2d.ace(scene_2, signal_to_embed)

        scene_1_ace_max_loc = np.where(
            scene_1_ace_result == np.max(scene_1_ace_result))
        scene_2_ace_max_loc = np.where(
            scene_2_ace_result == np.max(scene_2_ace_result))

        assert scene_2_ace_max_loc[0][0] == y_loc
        assert scene_2_ace_max_loc[1][0] == x_loc
        logging.debug("scene 2 target location identified correctly")
        assert scene_1_ace_max_loc[0][0] != y_loc or \
            scene_1_ace_max_loc[1][0] != x_loc
        logging.debug(
            "scene 1 target location was not identified "
            "correctly before covariance equalization"
        )

        scene_1_demeaned = sp2d.demean_image_data(scene_1)
        scene_2_demeaned = sp2d.demean_image_data(scene_2)

        scene_1_cov_equalized = sp2d.covariance_equalization_mean_centered(
            scene_1_demeaned, scene_2_demeaned)
        scene_1_equalized_ace_result = sp2d.ace(
            scene_1_cov_equalized, signal_to_embed)
        scene_1_equalized_ace_max_loc = np.where(
            scene_1_equalized_ace_result == np.max(
                scene_1_equalized_ace_result))
        assert scene_1_equalized_ace_max_loc[0][0] == y_loc
        assert scene_1_equalized_ace_max_loc[1][0] == x_loc
        logging.debug(
            "scene 1 target location was identified correctly "
            "after covariance equalization"
        )
        print("COVARIANCE EQUALIZATION TARGET DETECTION TEST PASSED")


if __name__ == '__main__':
    unittest.main()
