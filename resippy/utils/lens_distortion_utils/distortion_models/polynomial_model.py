import numpy
from numpy import ndarray


class PolynomialDistortionModel:
    """
    Polynomial model for lens distortion

    Image plane locations will be in the same units as what the focal length's assumed units are
    """
    def __init__(self,
                 polynomial_coefficients,  # type: ndarray
                 focal_length,  # type: float
                 ):
        self._polynomial_coefficients = polynomial_coefficients
        self._focal_length = focal_length

    def compute_undistorted_image_plane_locations(self,
                                                  distorted_x_image_points,  # type: ndarray
                                                  distorted_y_image_points,  # type: ndarray
                                                  ):  # type: (...) -> (ndarray, ndarray)
        distances = (distorted_y_image_points**2 + distorted_x_image_points**2)**0.5
        poly_terms = []
        for i, coeff in enumerate(self._polynomial_coefficients):
            poly_terms.append(distances**i * coeff)
        distortion = numpy.sum(numpy.squeeze(numpy.asarray(poly_terms)), axis=0)
        if len(distortion.shape) == 1:
            distortion = numpy.reshape(distortion, (len(distortion), 1))
        field_angles = numpy.arctan(distances/(self._focal_length * (distortion + 1)))
        undistorted_image_plane_distances = self._focal_length * numpy.tan(field_angles)
        distance_ratio = numpy.abs(undistorted_image_plane_distances / distances)
        undistorted_x_image_points = numpy.nan_to_num(distorted_x_image_points * distance_ratio)
        undistorted_y_image_points = numpy.nan_to_num(distorted_y_image_points * distance_ratio)
        return undistorted_x_image_points, undistorted_y_image_points
