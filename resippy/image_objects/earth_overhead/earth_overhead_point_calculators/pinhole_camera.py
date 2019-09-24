from __future__ import division

from resippy.utils import photogrammetry_utils
from resippy.utils.units import ureg

from numpy import ndarray


class PinholeCamera:
    """
    This is an idealized pinhole camera model.
    """
    # TODO find the photogrammetry book that was used to implement all of these calculations and cite it here.

    def __init__(self):
        # all of these units are in meters and radians
        self.X = None       # type: float
        self.Y = None       # type: float
        self.Z = None       # type: float
        self.omega = None   # type: float
        self.phi = None     # type: float
        self.kappa = None   # type: float
        self.f = None       # type: float
        self.M = None       # type: ndarray

        self.m11 = None
        self.m12 = None
        self.m13 = None

        self.m21 = None
        self.m22 = None
        self.m23 = None

        self.m31 = None
        self.m32 = None
        self.m33 = None

    def init_pinhole_from_coeffs(self,
                                 X,                         # type: float
                                 Y,                         # type: float
                                 Z,                         # type: float
                                 omega,                     # type: float
                                 phi,                       # type: float
                                 kappa,                     # type: float
                                 focal_length,              # type: float
                                 x_units='meters',          # type: str
                                 y_units='meters',          # type: str
                                 z_units='meters',          # type: str
                                 omega_units='radians',     # type: str
                                 phi_units='radians',       # type: str
                                 kappa_units='radians',     # type: str
                                 focal_length_units='mm'    # type: str
                                 ):                         # type: (...) -> None
        x = X * ureg.parse_expression(x_units)
        y = Y * ureg.parse_expression(y_units)
        z = Z * ureg.parse_expression(z_units)

        x_meters = x.to(ureg['meters'])
        y_meters = y.to(ureg['meters'])
        z_meters = z.to(ureg['meters'])

        omega_radians = omega * ureg.parse_expression(omega_units).to(ureg['radians'])
        phi_radians = phi * ureg.parse_expression(phi_units).to(ureg['radians'])
        kappa_radians = kappa * ureg.parse_expression(kappa_units).to(ureg['radians'])

        focal_length_meters = focal_length * ureg.parse_expression(focal_length_units).to(ureg['meters'])

        self.X = x_meters.magnitude
        self.Y = y_meters.magnitude
        self.Z = z_meters.magnitude
        self.omega = omega_radians.magnitude
        self.phi = phi_radians.magnitude
        self.kappa = kappa_radians.magnitude
        self.f = focal_length_meters.magnitude

        self.M = photogrammetry_utils.create_M_matrix(omega_radians, phi_radians, kappa_radians)

        self.m11 = self.M[0, 0]
        self.m12 = self.M[0, 1]
        self.m13 = self.M[0, 2]

        self.m21 = self.M[1, 0]
        self.m22 = self.M[1, 1]
        self.m23 = self.M[1, 2]

        self.m31 = self.M[2, 0]
        self.m32 = self.M[2, 1]
        self.m33 = self.M[2, 2]

    def world_to_image_plane(self,
                             world_x,  # type: ndarray
                             world_y,  # type: ndarray
                             world_z  # type: ndarray
                             ):        # type: (...) -> (ndarray, ndarray)
        """
        From the book Photogrammetry, third edition, by Francis H. Moffitt and Edward M. Mikhail,
        equation 6-15 on page 142
        :param world_x: 
        :param world_y: 
        :param world_z: 
        :return: 
        """
        x = -1.0*self.f*(
                (self.m11*(world_x - self.X) + self.m12*(world_y - self.Y) + self.m13*(world_z - self.Z)) /
                (self.m31*(world_x - self.X) + self.m32*(world_y - self.Y) + self.m33*(world_z - self.Z))
        )

        y = -1.0*self.f*(
                (self.m21*(world_x - self.X) + self.m22*(world_y - self.Y) + self.m23*(world_z - self.Z)) /
                (self.m31*(world_x - self.X) + self.m32*(world_y - self.Y) + self.m33*(world_z - self.Z))
        )

        return x, y
