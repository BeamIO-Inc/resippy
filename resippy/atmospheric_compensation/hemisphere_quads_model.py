import math

import numpy
from numpy import ndarray
from shapely.geometry import Polygon
import trimesh
import matplotlib.pyplot as plt
from PIL import Image
from skimage.draw import polygon as skimage_polygon


class HemisphereQuadsModel:
    def __init__(self):
        self.azimuth_angles = None  # type: ndarray
        self.elevation_angles = None  # type: ndarray
        self._quad_polygons = None
        self.center_xyz = None  # type: ndarray
        self._trimesh_model = None
        self._radius = 1
        self._uv_image = None  # type: ndarray

    @staticmethod
    def polar2cart(r, zenith, azimuth):
        cartesian_coords = [r * math.sin(zenith) * math.cos(azimuth),
                            r * math.sin(zenith) * math.sin(azimuth),
                            r * math.cos(zenith)
                            ]
        return cartesian_coords

    def _az_els_to_xyz(self, azimuths_angles, elevation_angles):
        zeniths = numpy.pi/2 - elevation_angles
        xyzs = numpy.zeros((len(azimuths_angles), 3))
        xyzs[:, 0] = self._radius * numpy.sin(zeniths) * numpy.cos(azimuths_angles)
        xyzs[:, 1] = self._radius * numpy.sin(zeniths) * numpy.sin(azimuths_angles)
        xyzs[:, 2] = self._radius * numpy.cos(zeniths)
        return xyzs

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
    def cap_polygon_radians(self):
        cap_azimuths = self.azimuth_angles
        cap_azimuths[-1] = cap_azimuths[0]
        cap_elevations = numpy.zeros_like(cap_azimuths) + self.elevation_angles[-1]
        az_els = []
        for az, el in zip(cap_azimuths, cap_elevations):
            az_els.append([az, el])
        cap = Polygon(az_els)
        return cap

    @property
    def cap_xyz_polygon(self):
        cap_radians = self.cap_polygon_radians
        cap_xyzs = self.az_el_polygon_to_xyz_polygon(cap_radians)
        return cap_xyzs

    @property
    def quad_az_el_vertices(self):
        az_vertices = numpy.tile(self.azimuth_angles[:-1], self.n_elevation_quads+1)
        el_vertices = numpy.repeat(self.elevation_angles, self.n_azimuth_quads)
        return az_vertices, el_vertices

    @property
    def all_az_el_vertices(self):
        az_vertices, el_vertices = self.quad_az_el_vertices
        return numpy.append(az_vertices, 0), numpy.append(el_vertices, numpy.pi/2)

    @property
    def quad_xyz_vertices(self):
        az, el = self.quad_az_el_vertices
        xyzs = self._az_els_to_xyz(az, el)
        return xyzs

    @property
    def quad_faces(self):
        quad_faces = []
        quad_face = numpy.array([0, 1, self.n_azimuth_quads+1, self.n_azimuth_quads])
        face_indices = numpy.tile(quad_face, self.n_azimuth_quads)
        face_indices = numpy.repeat(range(self.n_azimuth_quads), 4) + face_indices
        face_indices[-3] = quad_face[0]
        face_indices[-2] = quad_face[3]
        for elevation_index in range(self.n_elevation_quads):
            new_face_indices = face_indices + elevation_index * self.n_azimuth_quads
            quad_faces = quad_faces + list(new_face_indices)
        trimesh_quad_faces = numpy.reshape(numpy.array(quad_faces), (int(len(quad_faces) / 4), 4))
        return trimesh_quad_faces

    @property
    def n_azimuth_quads(self):
        return len(self.azimuth_angles) - 1

    @property
    def n_elevation_quads(self):
        return len(self.elevation_angles) - 1

    def create_trimesh_model(self):
        quad_xyz_vertices = self.quad_xyz_vertices
        quad_faces = self.quad_faces

        # get cap vertex and faces
        cap_vertex = self._az_els_to_xyz(numpy.array([0]), numpy.array([numpy.pi / 2]))
        all_xyz_vertices = numpy.append(quad_xyz_vertices, cap_vertex, axis=0)
        cap_vertex_index = len(all_xyz_vertices) - 1
        quad_xyz_vertices[-self.n_azimuth_quads:, :]
        top_index_end = len(quad_xyz_vertices)
        top_index_start = len(quad_xyz_vertices) - self.n_azimuth_quads

        tri_faces = trimesh.geometry.triangulate_quads(quad_faces)
        tri_faces = list(tri_faces)

        for i in numpy.arange(top_index_start, top_index_end-1):
            cap_face = [i, i+1, cap_vertex_index]
            tri_faces.append(cap_face)
        tri_faces.append([top_index_end-1, top_index_start, cap_vertex_index])

        hemisphere_trimesh = trimesh.Trimesh()

        hemisphere_trimesh.vertices = all_xyz_vertices
        hemisphere_trimesh.faces = tri_faces

        self._trimesh_model = hemisphere_trimesh

    @property
    def trimesh_model(self):
        return self._trimesh_model

    def az_el_to_uv_coords(self,
                           azimuths,
                           elevations):
        u0 = numpy.cos(azimuths)
        v0 = numpy.sin(azimuths)
        u = u0 * (numpy.pi/2 - elevations)/(numpy.pi/2)
        v = v0 * (numpy.pi/2 - elevations)/(numpy.pi/2)
        u = (1+u)/2.0
        v = (1+v)/2.0
        return u, v

    def uv_coords_to_pixel_yx_coords(self, u_coords, v_coords):
        nx, ny = self.uv_nx_ny
        x = u_coords * nx
        y = (1 - v_coords) * ny
        return y, x

    def color_quad(self,
                   az_index,  # type: int
                   el_index,  # type: int
                   rgb_color,  # type: [int, int, int]
                   ):
        az_el_polygon = self.get_quad_az_els_by_az_el_indices([az_index], [el_index])[0]
        az_coords, el_coords = az_el_polygon.boundary.coords.xy
        az_coords = numpy.array(az_coords)
        el_coords = numpy.array(el_coords)
        uv_coords = self.az_el_to_uv_coords(az_coords, el_coords)
        pixel_coords = self.uv_coords_to_pixel_yx_coords(uv_coords[0], uv_coords[1])
        rr, cc = skimage_polygon(pixel_coords[0], pixel_coords[1])
        self._uv_image[rr, cc, 0] = rgb_color[0]
        self._uv_image[rr, cc, 1] = rgb_color[1]
        self._uv_image[rr, cc, 2] = rgb_color[2]

    def color_cap(self,
                  rgb_color,  # type: [int, int, int]
                  ):
        cap_polygon = self.cap_polygon_radians
        az_coords, el_coords = cap_polygon.boundary.coords.xy
        az_coords = numpy.array(az_coords)
        el_coords = numpy.array(el_coords)
        uv_coords = self.az_el_to_uv_coords(az_coords, el_coords)
        pixel_coords = self.uv_coords_to_pixel_yx_coords(uv_coords[0], uv_coords[1])
        rr, cc = skimage_polygon(pixel_coords[0], pixel_coords[1])
        self._uv_image[rr, cc, 0] = rgb_color[0]
        self._uv_image[rr, cc, 1] = rgb_color[1]
        self._uv_image[rr, cc, 2] = rgb_color[2]

    @property
    def uv_nx_ny(self):
        ny, nx, bands = self._uv_image.shape
        return nx, ny

    def initialize_uv_image(self,
                            nx=1024,  # type: int
                            ny=1024,  # type; int
                            uv_base_color=[0, 0, 255]
                            ):
        self._uv_image = numpy.zeros((ny, nx, 3), dtype=numpy.uint8)
        self._uv_image[:, :, 0] = uv_base_color[0]
        self._uv_image[:, :, 1] = uv_base_color[1]
        self._uv_image[:, :, 2] = uv_base_color[2]

    def apply_uv_image(self):
        texture_image_data = numpy.array(self._uv_image)
        texture_pil_image = Image.fromarray(texture_image_data, 'RGB')

        texture_visual = trimesh.visual.TextureVisuals()
        texture_visual.material.image = texture_pil_image
        self.trimesh_model.visual = texture_visual
        az_vertices, el_vertices = self.all_az_el_vertices
        u_coords, v_coords = self.az_el_to_uv_coords(az_vertices, el_vertices)
        uv_coords = numpy.stack((u_coords, v_coords)).transpose()
        texture_visual.uv = uv_coords
        stop = 1
