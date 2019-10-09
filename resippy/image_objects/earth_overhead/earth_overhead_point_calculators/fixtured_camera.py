from __future__ import division

from resippy.utils import photogrammetry_utils
from resippy.utils.units import ureg

from numpy import ndarray


class FixturedCamera:

    def __init__(self):
        # all of these units are in meters and radians
        self.fixture_x = None                       # type: float
        self.fixture_y = None                       # type: float
        self.fixture_z = None                       # type: float

        self.fixture_M = None                       # type: ndarray

        self.x_rel_to_fixture = None  # type: float
        self.y_rel_to_fixture = None  # type: float
        self.z_rel_to_fixture = None  # type: float

        self.boresight_matrix = None  # type: ndarray

    def set_fixture_xyz(self,
                        x,          # type: float
                        y,          # type: float
                        z,          # type: float
                        ):
        self.fixture_x = x
        self.fixture_y = y
        self.fixture_z = z

    def set_fixture_orientation(self,
                                m_matrix,       # type: ndarray
                                ):
        self.fixture_M = m_matrix

    def set_fixture_orientation_by_roll_pitch_yaw(self,
                                                  roll,  # type: float
                                                  pitch,  # type: float
                                                  yaw,  # type: float
                                                  roll_units="radians",  # type: str
                                                  pitch_units="radians",  # type: str
                                                  yaw_units="radians",  # type: str
                                                  order='rpy',  # type: str
                                                  ):
        roll_radians = roll * ureg.parse_units(roll_units)
        pitch_radians = pitch * ureg.parse_units(pitch_units)
        yaw_radians = yaw * ureg.parse_units(yaw_units)

        roll_radians = roll_radians.to(ureg['radians'])
        pitch_radians = pitch_radians.to(ureg['radians'])
        yaw_radians = yaw_radians.to(ureg['radians'])

        m_matrix = photogrammetry_utils.create_M_matrix(roll_radians, pitch_radians, yaw_radians, order=order)
        self.set_fixture_orientation(m_matrix)


    def set_relative_camera_xyz(self,
                                x_relative_to_fixture_center,       # type: float
                                y_relative_to_fixture_center,       # type: float
                                z_relative_to_fixture_center,       # type: float
                                x_units='meters',
                                y_units='meters',
                                z_units='meters'
                                ):

        x = x_relative_to_fixture_center * ureg.parse_expression(x_units)
        y = y_relative_to_fixture_center * ureg.parse_expression(y_units)
        z = z_relative_to_fixture_center * ureg.parse_expression(z_units)

        x_meters = x.to(ureg['meters'])
        y_meters = y.to(ureg['meters'])
        z_meters = z.to(ureg['meters'])

        self.x_rel_to_fixture = x_meters.magnitude
        self.y_rel_to_fixture = y_meters.magnitude
        self.z_rel_to_fixture = z_meters.magnitude

    def set_boresight_matrix(self,
                             boresight_matrix,      # type: ndarray
                             ):
        self.boresight_matrix = boresight_matrix

    def set_boresight_matrix_from_camera_relative_rpy_params(self,
                                                             relative_roll,  # type: float
                                                             relative_pitch,  # type: float
                                                             relative_yaw,  # type: float
                                                             roll_units='radians',  # type: str
                                                             pitch_units='radians',  # type: str
                                                             yaw_units='radians',  # type; str
                                                             order='rpy',
                                                             ):
        roll_radians = relative_roll * ureg.parse_expression(roll_units)
        pitch_radians = relative_pitch * ureg.parse_expression(pitch_units)
        yaw_radians = relative_yaw * ureg.parse_expression(yaw_units)
        boresight_matrix = photogrammetry_utils.create_M_matrix(roll_radians, pitch_radians, yaw_radians, order=order)
        self.set_boresight_matrix(boresight_matrix)
        
    def get_fixture_xyz(self):
        return self.fixture_x, self.fixture_y, self.fixture_z

    def get_camera_absolute_xyz(self,
                                ):        # type: (...) -> tuple
        camera_point_in_space = [self.x_rel_to_fixture, self.y_rel_to_fixture, self.z_rel_to_fixture]
        rotated_camera_x, rotated_camera_y, rotated_camera_z = camera_point_in_space @ self.fixture_M
        absolute_camera_location = (self.fixture_x + rotated_camera_x, self.fixture_y + rotated_camera_y, self.fixture_z + rotated_camera_z)
        return absolute_camera_location

    def get_camera_absolute_M_matrix(self):
        """
        absolute orientation matrix
        :return:
        """
        return self.boresight_matrix @ self.fixture_M