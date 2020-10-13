from numpy import ndarray
from resippy.photogrammetry.lens_distortion_models.abstract_lens_distortion_model import AbstractLensDistortionModel
from resippy.utils.units.unit_constants import ureg


class PinholeDistortionModel(AbstractLensDistortionModel):
    """
    Pinhole model for lens distortion
    Image plane locations will be in the same units as what the focal length's assumed units are
    """

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
        x_points = (undistorted_x_image_points * ureg.parse_units(image_plane_units)).to('meters').magnitude
        y_points = (undistorted_y_image_points * ureg.parse_units(image_plane_units)).to('meters').magnitude

        return x_points, y_points
