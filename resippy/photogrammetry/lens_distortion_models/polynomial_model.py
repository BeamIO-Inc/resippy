import numpy
from numpy import ndarray
from resippy.photogrammetry.lens_distortion_models.abstract_lens_distortion_model import AbstractLensDistortionModel

from resippy.utils.units.unit_constants import ureg


class PolynomialDistortionModel(AbstractLensDistortionModel):
    """
    Polynomial model for lens distortion
    Image plane locations will be in the same units as what the focal length's assumed units are
    """
    def __init__(self,
                 polynomial_coefficients,  # type: ndarray
                 focal_length,  # type: float
                 focal_length_units,  # type: float
                 ):
        self._polynomial_coefficients = polynomial_coefficients
        self._focal_length = focal_length
        self._focal_length_units = focal_length_units

    def compute_distorted_image_plane_locations(self,
                                                undistorted_x_image_points,  # type: ndarray
                                                undistorted_y_image_points,  # type: ndarray
                                                image_plane_units,  # type: str
                                                ):  # type: (...) -> (ndarray, ndarray)
        """

        :param undistorted_x_image_points: 2d array of "x" image plane locations
        :param undistorted_y_image_points: 2d array of "y" image plane locations
        :param image_plane_units: str defining image plane units
        :return: transformed x and y image plane locations caused by lens distortion effects.

        The input of the model should be aware of units.  Coefficients are dependent on the input units of the
        the focal length, but the output units of this method will always be in meters.

        """
        undistorted_x_image_points = (undistorted_x_image_points * ureg.parse_units(image_plane_units)).to(self._focal_length_units).magnitude
        undistorted_y_image_points = (undistorted_y_image_points * ureg.parse_units(image_plane_units)).to(self._focal_length_units).magnitude

        distances = (undistorted_y_image_points ** 2 + undistorted_x_image_points ** 2) ** 0.5
        poly_terms = []
        for i, coeff in enumerate(self._polynomial_coefficients):
            poly_terms.append(distances**i * coeff)
        distortion = numpy.sum(numpy.squeeze(numpy.asarray(poly_terms)), axis=0)
        if len(distortion.shape) == 1:
            distortion = numpy.reshape(distortion, (len(distortion), 1))
        field_angles = numpy.arctan(distances/(self._focal_length * (distortion + 1)))
        undistorted_image_plane_distances = self._focal_length * numpy.tan(field_angles)
        distance_ratio = numpy.abs(undistorted_image_plane_distances / distances)
        distorted_x_image_points = numpy.nan_to_num(undistorted_x_image_points * distance_ratio)
        distorted_y_image_points = numpy.nan_to_num(undistorted_y_image_points * distance_ratio)

        distorted_x_image_points = (distorted_x_image_points * ureg.parse_units(self._focal_length_units)).to('meters').magnitude
        distorted_y_image_points = (distorted_y_image_points * ureg.parse_units(self._focal_length_units)).to('meters').magnitude

        return distorted_x_image_points, distorted_y_image_points
