import cv2 as cv

import numpy

import copy
from resippy.image_objects.earth_overhead.physical_camera.physical_camera_image import PhysicalCameraImage
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.ideal_pinhole_fpa_local_utm_point_calc import \
    IdealPinholeFpaLocalUtmPointCalc
from resippy.image_objects.earth_overhead.abstract_earth_overhead_image import AbstractEarthOverheadImage
from resippy.image_objects.earth_overhead.igm.igm_image import IgmImage
from resippy.utils import photogrammetry_utils
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.fixtured_camera import FixturedCamera
from resippy.photogrammetry.dem.abstract_dem import AbstractDem
from resippy.photogrammetry import ortho_tools
from resippy.utils.image_utils import image_utils
from resippy.utils import numpy_and_array_utils
from resippy.photogrammetry.dem.dem_factory import DemFactory
import scipy.stats as stats
import matplotlib.pyplot as plt


class SiftBoresighter:
    def __init__(self,
                 dem=None,  # type: AbstractDem
                 ):
        self._dem = dem
        if dem is None:
            self._dem = DemFactory.constant_elevation(0)

    def get_image_object(self,
                         index,  # type: int
                         ):  # type: (...) -> PhysicalCameraImage
        return self._image_objects[index]

    def compute_ground_metrics(self, image1_lons, image1_lats, image2_lons, image2_lats):
        lon_diffs = image1_lons - image2_lons
        lat_diffs = image1_lats - image2_lats
        distances = (lat_diffs ** 2 + lon_diffs ** 2) ** 0.5

        average_lon_diff = numpy.average(lon_diffs)
        average_lat_diff = numpy.average(lat_diffs)
        average_distance = numpy.average(distances)

        return lon_diffs, lat_diffs, distances, average_lon_diff, average_lat_diff, average_distance

    def compute_antiparallel_flightline_image_metrics(self,
                                                      image_obj1,  # type: PhysicalCameraImage
                                                      image_obj2,  # type: PhysicalCameraImage
                                                      image1_lons,  # type: numpy.ndarray
                                                      image1_lats,  # type: numpy.ndarray
                                                      image2_lons,  # type: numpy.ndarray
                                                      image2_lats,  # type: numpy.ndarray
                                                      ):
        im1_elevations = self._dem.get_elevations(image1_lons,
                                                  image1_lats,
                                                  world_proj=image_obj1.pointcalc.get_projection())

        im2_elevations = self._dem.get_elevations(image2_lons,
                                                  image2_lats,
                                                  world_proj=image_obj2.pointcalc.get_projection())

        image1_x_points, image1_y_points = image_obj1.pointcalc.lon_lat_alt_to_pixel_x_y(image1_lons,
                                                                                         image1_lats,
                                                                                         im1_elevations)
        # image2_x_points, image2_y_points = image_obj2.pointcalc.lon_lat_alt_to_pixel_x_y(image2_lons,
        #                                                                                  image2_lats,
        #                                                                                  im2_elevations)

        x1, y1 = image_obj1.pointcalc.lon_lat_alt_to_pixel_x_y(image2_lons, image2_lats, numpy.zeros_like(image2_lons))
        # x2, y2 = image_obj2.pointcalc.lon_lat_alt_to_pixel_x_y(image1_lons, image1_lats, numpy.zeros_like(image1_lons))

        x_ifov_diffs = image1_x_points - x1
        y_ifov_diffs = image1_y_points - y1
        ifov_diffs = (x_ifov_diffs ** 2 + y_ifov_diffs ** 2) ** 0.5

        average_x_ifov_diff = numpy.average(x_ifov_diffs)
        average_y_ifov_diff = numpy.average(y_ifov_diffs)
        average_ifov_diffs = numpy.average(ifov_diffs)

        return y_ifov_diffs, x_ifov_diffs, ifov_diffs, average_y_ifov_diff, average_x_ifov_diff, average_ifov_diffs

    def remove_outliers_using_iqr(self,
                                  image1_x_matches,
                                  image1_y_matches,
                                  image2_x_matches,
                                  image2_y_matches,
                                  n_iqr_from_first_and_third_quartiles=1.5):
        x_diffs = image2_x_matches - image1_x_matches
        y_diffs = image2_y_matches - image1_y_matches
        x_iqr = stats.iqr(x_diffs)
        y_iqr = stats.iqr(y_diffs)
        x_median = numpy.median(x_diffs)
        y_median = numpy.median(y_diffs)
        x_low = x_median - x_iqr * (0.5 + n_iqr_from_first_and_third_quartiles)
        x_high = x_median + x_iqr * (0.5 + n_iqr_from_first_and_third_quartiles)
        y_low = y_median - y_iqr * (0.5 + n_iqr_from_first_and_third_quartiles)
        y_high = y_median + y_iqr * (0.5 + n_iqr_from_first_and_third_quartiles)
        good_indices = numpy.where(x_diffs < x_high) or \
                       numpy.where(x_diffs > x_low) or \
                       numpy.where(y_diffs < y_high) or \
                       numpy.where(y_diffs > y_low)
        return image1_x_matches[good_indices], \
               image1_y_matches[good_indices], \
               image2_x_matches[good_indices], \
               image2_y_matches[good_indices]

    def get_im1_im2_lon_lat_matches(self,
                                    image_object1,
                                    image_object2,
                                    ):
        img1_ortho = ortho_tools.create_full_ortho_gtiff_image(image_object1, dem=self._dem)
        img2_ortho = ortho_tools.create_full_ortho_gtiff_image(image_object2, dem=self._dem)

        gtiff1 = img1_ortho.get_image_band(0)
        gtiff2 = img2_ortho.get_image_band(0)

        gtiff1 = image_utils.normalize_grayscale_image(gtiff1, 0, 255, numpy.uint8)
        gtiff2 = image_utils.normalize_grayscale_image(gtiff2, 0, 255, numpy.uint8)

        # Initiate SIFT detector
        sift = cv.xfeatures2d_SIFT.create()
        # find the keypoints and descriptors with SIFT

        kp1, des1 = sift.detectAndCompute(gtiff1, None)
        kp2, des2 = sift.detectAndCompute(gtiff2, None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        gtiff1_xy_matches = []
        gtiff2_xy_matches = []

        # ratio test as per Lowe's paper
        for i, (first_match, second_match) in enumerate(matches):
            if first_match.distance < 0.7 * second_match.distance:
                image1_keypoint_index = first_match.queryIdx
                image2_keypoint_index = first_match.trainIdx

                gtiff1_xy_matches.append(kp1[image1_keypoint_index].pt)
                gtiff2_xy_matches.append(kp2[image2_keypoint_index].pt)

        gtiff1_x_matches, gtiff1_y_matches = numpy_and_array_utils.list_of_tuples_to_separate_lists(gtiff1_xy_matches)
        gtiff2_x_matches, gtiff2_y_matches = numpy_and_array_utils.list_of_tuples_to_separate_lists(gtiff2_xy_matches)

        gtiff1_x_matches = numpy.asarray(gtiff1_x_matches)
        gtiff1_y_matches = numpy.asarray(gtiff1_y_matches)
        gtiff2_x_matches = numpy.asarray(gtiff2_x_matches)
        gtiff2_y_matches = numpy.asarray(gtiff2_y_matches)

        image1_lons, image1_lats = img1_ortho.pointcalc.pixel_x_y_alt_to_lon_lat(gtiff1_x_matches,
                                                                                 gtiff1_y_matches,
                                                                                 numpy.zeros_like(gtiff1_x_matches))
        image2_lons, image2_lats = img2_ortho.pointcalc.pixel_x_y_alt_to_lon_lat(gtiff2_x_matches,
                                                                                 gtiff2_y_matches,
                                                                                 numpy.zeros_like(gtiff2_x_matches))
        filtered_lon_lat_matches = self.remove_outliers_using_iqr(image1_lons,
                                                                  image1_lats,
                                                                  image2_lons,
                                                                  image2_lats)
        image1_lons = filtered_lon_lat_matches[0]
        image1_lats = filtered_lon_lat_matches[1]
        image2_lons = filtered_lon_lat_matches[2]
        image2_lats = filtered_lon_lat_matches[3]
        return image1_lons, image1_lats, image2_lons, image2_lats

    def create_boresight_corrected_image_object(self,
                                                input_image_object,  # type: PhysicalCameraImage
                                                boresight_roll,  # type: float
                                                boresight_pitch,  # type: float
                                                boresight_yaw,  # type: float
                                                boresight_units="degrees",  # type: str
                                                order="rpy",  # type: str
                                                ):
        new_image_obj = copy.deepcopy(input_image_object)
        new_point_calc = self.create_new_camera_model(input_image_object.pointcalc,
                                                      boresight_roll,
                                                      boresight_pitch,
                                                      boresight_yaw,
                                                      boresight_units=boresight_units,
                                                      order=order)
        new_image_obj.pointcalc = new_point_calc
        return new_image_obj

    def create_new_camera_model(self,
                                old_camera_model,  # type: IdealPinholeFpaLocalUtmPointCalc
                                boresight_roll,  # type: float
                                boresight_pitch,  # type: float
                                boresight_yaw,  # type: float
                                boresight_units="degrees",  # type: str
                                order="rpy",  # type: str
                                ):
        fixtured_camera = FixturedCamera()

        fixtured_camera.set_boresight_matrix_from_camera_relative_rpy_params(boresight_roll,
                                                                             boresight_pitch,
                                                                             boresight_yaw,
                                                                             roll_units=boresight_units,
                                                                             pitch_units=boresight_units,
                                                                             yaw_units=boresight_units,
                                                                             order=order)

        fixtured_camera.set_fixture_orientation_by_roll_pitch_yaw(old_camera_model._pinhole_camera.omega,
                                                                  old_camera_model._pinhole_camera.phi,
                                                                  old_camera_model._pinhole_camera.kappa)

        m_matrix = fixtured_camera.get_camera_absolute_M_matrix()
        omega, phi, kappa = photogrammetry_utils.solve_for_omega_phi_kappa(m_matrix)
        if numpy.isnan(kappa):
            kappa = numpy.pi - 0.00000000001

        point_calc = IdealPinholeFpaLocalUtmPointCalc.init_from_local_params(old_camera_model._pinhole_camera.X,
                                                                             old_camera_model._pinhole_camera.Y,
                                                                             old_camera_model._pinhole_camera.Z,
                                                                             old_camera_model.get_projection(),
                                                                             omega,
                                                                             phi,
                                                                             kappa,
                                                                             old_camera_model._npix_x,
                                                                             old_camera_model._npix_y,
                                                                             old_camera_model._pixel_pitch_x_meters,
                                                                             old_camera_model._pixel_pitch_y_meters,
                                                                             old_camera_model._pinhole_camera.f,
                                                                             alt_units="meters",
                                                                             pixel_pitch_x_units="meters",
                                                                             pixel_pitch_y_units="meters",
                                                                             focal_length_units="meters",
                                                                             flip_x=old_camera_model._flip_x,
                                                                             flip_y=old_camera_model._flip_y)

        return point_calc

    def compute_search_space_rolls_and_pitches(self,
                                               image_obj1,  # type: PhysicalCameraImage
                                               image_obj2,  # type: PhysicalCameraImage
                                               n_ifov_x_search=20,  # type: int
                                               n_ifov_y_search=20,  # type: int
                                               ifov_resolution=1,  # type: float
                                               ):
        image1_lons, image1_lats, image2_lons, image2_lats = self.get_im1_im2_lon_lat_matches(image_obj1,
                                                                                              image_obj2)
        image_metrics = self.compute_antiparallel_flightline_image_metrics(image_obj1,
                                                                           image_obj2,
                                                                           image1_lons,
                                                                           image1_lats,
                                                                           image2_lons,
                                                                           image2_lats)

        ifov_x = image_obj1.pointcalc.compute_ifov_x(output_units="radians")
        ifov_y = image_obj1.pointcalc.compute_ifov_y(output_units="radians")
        rough_boresight_roll = image_metrics[3] * ifov_x / 2
        rough_boresight_pitch = image_metrics[4] * ifov_y / 2

        boresight_roll_start = rough_boresight_roll - ifov_x * n_ifov_x_search / 2
        boresight_roll_end = rough_boresight_roll + ifov_x * n_ifov_x_search / 2
        boresight_pitch_start = rough_boresight_pitch - ifov_y * n_ifov_y_search / 2
        boresight_pitch_end = rough_boresight_pitch + ifov_y * n_ifov_y_search / 2

        boresight_rolls = numpy.linspace(boresight_roll_start, boresight_roll_end,
                                         int(n_ifov_x_search / ifov_resolution))
        boresight_pitches = numpy.linspace(boresight_pitch_start, boresight_pitch_end,
                                           int(n_ifov_x_search / ifov_resolution))

        boresight_rolls = boresight_rolls + ifov_x * 0.000001
        boresight_pitches = boresight_pitches + ifov_y * 0.000001

        return boresight_rolls, boresight_pitches

    def create_boresight_detection_planes(self,
                                          image_obj1,  # type: PhysicalCameraImage
                                          image_obj2,  # type: PhysicalCameraImage
                                          boresight_rolls,  # type: numpy.ndarray
                                          boresight_pitches,  # type: numpy.ndarray
                                          boresight_yaws,  # type: numpy.ndarray
                                          ):
        """
        Assumes input images have physical camera models with zero boresights

        :param image_obj1:
        :param image_obj2:
        :return:
        """
        image1_lons, image1_lats, image2_lons, image2_lats = self.get_im1_im2_lon_lat_matches(image_obj1,
                                                                                              image_obj2)
        image1_x_pixels, image1_y_pixels = image_obj1.pointcalc.lon_lat_alt_to_pixel_x_y(image1_lons,
                                                                                         image1_lats,
                                                                                         numpy.zeros_like(image1_lons))
        image2_x_pixels, image2_y_pixels = image_obj2.pointcalc.lon_lat_alt_to_pixel_x_y(image2_lons,
                                                                                         image2_lats,
                                                                                         numpy.zeros_like(image2_lons))

        boresight_detection_planes = []

        if boresight_yaws is None:
            boresight_yaws = [0]

        for boresight_yaw in boresight_yaws:
            boresight_optimization_grid = numpy.zeros((len(boresight_pitches), len(boresight_rolls)))
            for r_i, boresight_roll in enumerate(boresight_rolls):
                print(str(r_i / len(boresight_rolls)))
                for p_i, boresight_pitch in enumerate(boresight_pitches):
                    new_pointcalc1 = self.create_new_camera_model(image_obj1.pointcalc,
                                                                  boresight_roll,
                                                                  boresight_pitch,
                                                                  boresight_yaw,
                                                                  boresight_units="radians")

                    new_pointcalc2 = self.create_new_camera_model(image_obj2.pointcalc,
                                                                  boresight_roll,
                                                                  boresight_pitch,
                                                                  boresight_yaw,
                                                                  boresight_units="radians")

                    new_lons1, new_lats1 = new_pointcalc1.pixel_x_y_alt_to_lon_lat(image1_x_pixels, image1_y_pixels,
                                                                                   numpy.zeros_like(image1_x_pixels))
                    new_lons2, new_lats2 = new_pointcalc2.pixel_x_y_alt_to_lon_lat(image2_x_pixels, image2_y_pixels,
                                                                                   numpy.zeros_like(image2_x_pixels))

                    new_ground_metrics = self.compute_ground_metrics(new_lons1, new_lats1, new_lons2, new_lats2)

                    boresight_optimization_grid[p_i, r_i] = new_ground_metrics[5]
            boresight_detection_planes.append(boresight_optimization_grid)
        return boresight_detection_planes
