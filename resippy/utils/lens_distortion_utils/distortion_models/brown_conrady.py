import numpy
from numpy import ndarray


class BrownConradyDistortionModel:
    """
    Brown-Conrady distortion model from:
    equation 1 of the following paper:
    http://researchspace.csir.co.za/dspace/bitstream/handle/10204/3168/De%20Villiers_2008.pdf
    """
    def __init__(self,
                 x_center,  # type: float
                 y_center,  # type: float
                 radial_distortion_coefficients,  # type: ndarray
                 tangential_distortion_coefficient,  # type: ndarray
                 ):
        self._xc = x_center
        self._yc = y_center
        self._k = radial_distortion_coefficients
        self._p = tangential_distortion_coefficient

    def compute_undistorted_image_plane_locations(self,
                                                  distorted_x_image_points,  # type: ndarray
                                                  distorted_y_image_points,  # type: ndarray
                                                  ):  # type: (...) -> (ndarray, ndarray)
        r_squared = (distorted_x_image_points - self._xc)**2 + (distorted_y_image_points - self._yc)**2
        n_r_terms = numpy.max((len(self._k), len(self._p)))
        n_k_terms = len(self._k)
        n_p_terms = len(self._p)
        r_terms = [r_squared**(i+1) for i in range(n_r_terms)]

        kr_terms = [self._k[i] * r_terms[i] for i in range(n_k_terms)]
        first_term = numpy.sum(kr_terms, axis=0)
        second_term_x = self._p[0] * (r_squared + 2*(distorted_x_image_points - self._xc)**2) + \
                        2*self._p[1]*(distorted_x_image_points - self._xc)*(distorted_y_image_points - self._yc)
        second_term_y = self._p[1] * (r_squared + 2*(distorted_y_image_points - self._yc)**2) + \
                        2*self._p[0]*(distorted_x_image_points - self._xc)*(distorted_y_image_points - self._yc)
        pr_terms = [self._p[i+2] * r_terms[i] for i in range(n_p_terms-2)]
        third_term = 1 + numpy.sum(pr_terms, axis=0)

        x_u = distorted_x_image_points + (distorted_x_image_points - self._xc)*first_term + second_term_x * third_term
        y_u = distorted_y_image_points + (distorted_y_image_points - self._yc)*first_term + second_term_y * third_term

        return x_u, y_u
