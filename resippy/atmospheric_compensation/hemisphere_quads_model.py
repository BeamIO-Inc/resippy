import math

import numpy
from numpy import ndarray
from shapely.geometry import Polygon


class HemisphereQuadsModel:
    def __init__(self):
        self.azimuth_angles = None  # type: ndarray
        self.elevation_angles = None  # type: ndarray
        self._quad_polygons = None
        self.center_xyz = None  # type: ndarray
        self._radius = 1

    @staticmethod
    def polar2cart(r, zenith, azimuth):
        cartesian_coords = [r * math.sin(zenith) * math.cos(azimuth),
                            r * math.sin(zenith) * math.sin(azimuth),
                            r * math.cos(zenith)
                            ]
        return cartesian_coords

    @staticmethod
    def equal_area_elevation_angles(n_elevation_quads,  # type: int
                                    max_elevation,  # type: float
                                    max_elevation_units='radians',  # type: str
                                    ):
        if max_elevation_units == 'degrees':
            max_elevation = numpy.deg2rad(max_elevation)
        zenith_min = numpy.pi/2 - max_elevation
        total_area = (-numpy.cos(numpy.pi/2.0) + numpy.cos(zenith_min))
        each_area = total_area / n_elevation_quads
        zenith_angles = [zenith_min]
        for quad_num in range(n_elevation_quads):
            theta_n = zenith_angles[-1]
            theta_n_plus_one = numpy.arccos(numpy.cos(theta_n) - each_area)
            zenith_angles.append(theta_n_plus_one)
        elevation_angles = numpy.pi/2 - numpy.asarray(zenith_angles)
        elevation_angles = numpy.flip(elevation_angles)
        return elevation_angles

    @classmethod
    def create_from_equal_az_el_spacings(cls,
                                         n_azimuths,  # type: int
                                         n_elevations,  # type: int
                                         max_elevation_degrees,  # type: int
                                         ):
        hemisphere = cls()
        hemisphere.azimuth_angles = numpy.linspace(0, 2 * numpy.pi, n_azimuths + 1)
        hemisphere.elevation_angles = numpy.linspace(0, numpy.deg2rad(max_elevation_degrees), n_elevations + 1)
        return hemisphere

    @classmethod
    def create_from_equal_areas(cls,
                                n_azimuths,  # type: int
                                n_elevations,  # type: int
                                max_elevation_degrees,  # type: int
                                ):
        hemisphere = cls()
        elevation_angles = cls.equal_area_elevation_angles(n_elevations,
                                                           max_elevation_degrees,
                                                           max_elevation_units='degrees')
        hemisphere.azimuth_angles = numpy.linspace(0, 2 * numpy.pi, n_azimuths + 1)
        hemisphere.elevation_angles = elevation_angles
        return hemisphere

    def get_quad_az_els_by_az_el_indices(self,
                                         azimuth_indices,  # type: int
                                         elevation_indices,  # type: int
                                         units='radians',
                                         ):
        # TODO: put in an out of range exception here, check if azimuth and elevation indices are within range
        # TODO: catch an exception for units, should only be 'radians' or 'degrees'
        polygons = []
        for az_index, el_index in zip(azimuth_indices, elevation_indices):
            az_0 = self.azimuth_angles[az_index]
            az_1 = self.azimuth_angles[az_index + 1]
            el_0 = self.elevation_angles[el_index]
            el_1 = self.elevation_angles[el_index + 1]
            if units.lower() == 'degrees':
                az_0 = numpy.rad2deg(az_0)
                az_1 = numpy.rad2deg(az_1)
                el_0 = numpy.rad2deg(el_0)
                el_1 = numpy.rad2deg(el_1)
            poly = Polygon([[az_0, el_0], [az_1, el_0], [az_1, el_1], [az_0, el_1], [az_0, el_0]])
            polygons.append(poly)
        return polygons

    def az_el_polygon_to_xyz_polygon(self,
                                     polygon,  # type: Polygon
                                     ):
        # TODO: put in an exception if the center_xyz has not been set
        angles = polygon.exterior.coords
        az_el_coords = [a for a in angles]
        xyzs = []
        for az_el in az_el_coords:
            xyz = self.polar2cart(self._radius, numpy.pi / 2 - az_el[1], az_el[0])
            xyz = numpy.array(xyz) + numpy.array(self.center_xyz)
            xyzs.append(list(xyz))
        poly = Polygon(xyzs)
        return poly

    def get_quad_xyzs_by_az_el_indices(self,
                                       azimuth_indices,  # type: [int]
                                       elevation_indices,  # type: [int]
                                       ):
        # TODO: put in an out of range exception here, check if azimuth and elevation indices are within range
        # TODO: put in an exception if the center xyz coordinate hasn't been set
        polygons = self.get_quad_az_els_by_az_el_indices(azimuth_indices, elevation_indices, units='radians')
        xyz_polys = []
        for poly in polygons:
            xyz_poly = self.az_el_polygon_to_xyz_polygon(poly)
            xyz_polys.append(xyz_poly)
        return xyz_polys

    @property
    def all_quad_az_els_degrees(self):
        azimuth_indices = numpy.tile(numpy.arange(0, len(self.azimuth_angles) - 1), len(self.elevation_angles) - 1)
        elevation_indices = numpy.repeat(numpy.arange(0, len(self.elevation_angles) - 1), len(self.azimuth_angles) - 1)
        quads = self.get_quad_az_els_by_az_el_indices(azimuth_indices, elevation_indices, units='degrees')
        return quads

    @property
    def all_quad_az_els_radians(self):
        azimuth_indices = numpy.tile(numpy.arange(0, len(self.azimuth_angles) - 1), len(self.elevation_angles) - 1)
        elevation_indices = numpy.repeat(numpy.arange(0, len(self.elevation_angles) - 1), len(self.azimuth_angles) - 1)
        quads = self.get_quad_az_els_by_az_el_indices(azimuth_indices, elevation_indices, units='radians')
        return quads

    @property
    def all_quad_xyzs(self):
        azimuth_indices = numpy.tile(numpy.arange(0, len(self.azimuth_angles) - 1), len(self.elevation_angles) - 1)
        elevation_indices = numpy.repeat(numpy.arange(0, len(self.elevation_angles) - 1), len(self.azimuth_angles) - 1)
        quads = self.get_quad_xyzs_by_az_el_indices(azimuth_indices, elevation_indices)
        return quads

    @property
    def cap_az_els_degrees(self):
        cap_azimuths = self.azimuth_angles
        cap_azimuths[-1] = cap_azimuths[0]
        cap_azimuths = numpy.rad2deg(cap_azimuths)
        cap_elevations = numpy.zeros_like(cap_azimuths) + numpy.rad2deg(self.elevation_angles[-1])
        az_els = []
        for az, el in zip(cap_azimuths, cap_elevations):
            az_els.append([az, el])
        cap = Polygon(az_els)
        return cap

    @property
    def cap_az_els_radians(self):
        cap_azimuths = self.azimuth_angles
        cap_azimuths[-1] = cap_azimuths[0]
        cap_elevations = numpy.zeros_like(cap_azimuths) + self.elevation_angles[-1]
        az_els = []
        for az, el in zip(cap_azimuths, cap_elevations):
            az_els.append([az, el])
        cap = Polygon(az_els)
        return cap

    @property
    def cap_xyzs(self):
        cap_radians = self.cap_az_els_radians
        cap_xyzs = self.az_el_polygon_to_xyz_polygon(cap_radians)
        return cap_xyzs
