from __future__ import division

import abc
from six import add_metaclass


@add_metaclass(abc.ABCMeta)
class AbstractLensDistortionModel:
    """
    This defines the abstract lens distortion model for the purposes of photogrammetry and projecting points from
    the ground to a focal plane through a lens model.  The only main rule for this abstraction is that a concrete
    implementation must have a method named "compute_distorted_image_plane_locations", which converts known pixel
    locations on an image plane and transforms them into warped locations as seen by the lens model.
    """

    @abc.abstractmethod
    def compute_distorted_image_plane_locations(self,
                                                undistorted_x_image_points,  # type: ndarray
                                                undistorted_y_image_points,  # type: ndarray
                                                image_plane_units,  # type: str
                                                ):  # type: (...) -> ndarray
        """
        transforms known pixel locations on an image plane and transforms them to warped locations as seen by the lens
        model.  The output of this can later be used by classes in the earth_overhead_point_calculators directory,
        such as pinhole_camera, line_scanner_point_calc, or fpa_distortion_mapped_point_calc
        """
