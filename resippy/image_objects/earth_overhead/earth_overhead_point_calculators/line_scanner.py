from __future__ import division

from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.abstract_earth_overhead_point_calc import AbstractEarthOverheadPointCalc
from resippy.photogrammetry import crs_defs
from resippy.utils import proj_utils
from pyproj import transform
from pyproj import Proj
from resippy.utils.units import ureg
from numpy import ndarray
import numpy as np
from typing import Union


class LineScannerPointCalc(AbstractEarthOverheadPointCalc):

    def __init__(self):
        super(LineScannerPointCalc, self).__init__()
        self.sensor_line_utm_eastings = None                 # type: [float]
        self.sensor_line_utm_northings = None                  # type: [float]
        self.sensor_line_utm_alts = None                   # type: [float]
        self.sensor_line_m_matrices = None                 # type: [ndarray]
        self.pixel_cross_track_angles_radians = None                 # type: [float]
        self.pixel_along_track_angles_radians = None                 # type: [float]
        self.boresight_m_matrix = np.eye(3)                          # type: ndarray

    def _pixel_x_y_alt_to_lon_lat_native(self, pixel_xs, pixel_ys, alts=None, band=None):
        pass

    def _lon_lat_alt_to_pixel_x_y_native(self, lons, lats, alts, band=None):
        pass

    def set_line_utm_eastings(self, eastings):
        self.sensor_line_utm_eastings = eastings

    def set_line_utm_northings(self, northings):
        self.sensor_line_utm_northings = northings

    def set_line_alts(self, alts):
        self.sensor_line_utm_alts = alts

    def set_line_m_matrices(self,
                            m_matrices,             # type: list
                            ):
        self.sensor_line_m_matrices = m_matrices

    def set_pixel_cross_track_angles(self,
                                     angles,                        # type: ndarray
                                     angle_units='radians'          # type: str
                                     ):

        angles_radians = angles * ureg.parse_expression(angle_units)
        angles_radians = angles_radians.to('radians').magnitude
        self.pixel_cross_track_angles_radians = angles_radians

    def set_pixel_along_track_angles(self,
                                     angles,                        # type: Union[ndarray, float]
                                     angle_units='radians'          # type: str
                                     ):
        if not type(angles)==type(np.array([0, 0])):
            angles = np.zeros_like(self.pixel_cross_track_angles_radians) + angles
        angles_radians = angles * ureg.parse_expression(angle_units)
        angles_radians = angles_radians.to('radians').magnitude
        self.pixel_along_track_angles_radians = angles_radians

    def set_boresight_matrix(self,
                             boresight_matrix,          # type: ndarray
                             ):
        self.boresight_m_matrix = boresight_matrix

    def get_all_pixel_sensor_yxzs(self):
        ny = len(self.sensor_line_utm_northings)
        nx = len(self.pixel_cross_track_angles_radians)
        all_yxzs = np.zeros((ny, nx, 3))
        for i in range(nx):
            all_yxzs[:, i, 1] = self.sensor_line_utm_eastings
            all_yxzs[:, i, 0] = self.sensor_line_utm_northings
            all_yxzs[:, i, 2] = self.sensor_line_utm_alts
        return all_yxzs

    def get_all_pixel_vectors_xyz(self):
        ny = len(self.sensor_line_utm_northings)
        nx = len(self.pixel_cross_track_angles_radians)
        all_sensor_vecs = np.zeros((ny, nx, 3))
        local_pixel_yxzs = np.zeros((nx, 3))
        local_pixel_yxzs[:, 1] = np.sin(self.pixel_cross_track_angles_radians)
        local_pixel_yxzs[:, 0] = np.sin(self.pixel_along_track_angles_radians)
        local_pixel_yxzs[:, 2] = np.ones(nx)
        # bump the vectors over by the boresight angles
        for i in range(nx):
            local_pixel_yxzs[i, :] = np.matmul(self.boresight_m_matrix, local_pixel_yxzs[i, :])
        for i, line_matrix in enumerate(self.sensor_line_m_matrices):
            line_vecs = line_matrix @ local_pixel_yxzs.transpose()
            all_sensor_vecs[i, :, :] = line_vecs.transpose()
            all_sensor_vecs[i, :, 0] = all_sensor_vecs[i, :, 0] / all_sensor_vecs[0, :, 2]
            all_sensor_vecs[i, :, 1] = all_sensor_vecs[i, :, 1] / all_sensor_vecs[0, :, 2]
            all_sensor_vecs[i, :, 2] = all_sensor_vecs[i, :, 2] / all_sensor_vecs[0, :, 2]

        return all_sensor_vecs

    def get_all_world_xy_coords(self,
                                alts=None
                                ):
        yxzs = self.get_all_pixel_sensor_yxzs()
        vecs = self.get_all_pixel_vectors_xyz()
        if alts is None:
            alts = np.zeros_like(vecs[:, :, 2])
        elif type(alts) is not type(np.array(0)):
            alts = np.zeros_like(vecs[:, :, 2]) + alts
        agl_alts = yxzs[:, :, 2] - alts
        yxzs[:, :, 2] = agl_alts
        world_x_coords = vecs[:, :, 1] * agl_alts + yxzs[:, :, 1]
        world_y_coords = vecs[:, :, 0] * agl_alts + yxzs[:, :, 0]

        return world_x_coords, world_y_coords
