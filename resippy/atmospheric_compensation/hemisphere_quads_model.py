import numpy
from numpy import ndarray
from shapely.geometry import Polygon
import trimesh
from resippy.utils import coordinate_conversions
from resippy.utils import photogrammetry_utils
from PIL import Image
from skimage.draw import polygon as skimage_polygon
from resippy.atmospheric_compensation.utils import hemisphere_coordinate_conversions


class HemisphereQuadsModel:
    def __init__(self):
        self.azimuth_angles = None  # type: ndarray
        self.elevation_angles = None  # type: ndarray
        self._quad_polygons = None
        self.center_xyz = None  # type: ndarray
        self._trimesh_model = None
        self._radius = 1
        self._uv_image = None  # type: ndarray
        self._sun_size_degrees = 0.5  # type: float
        self._sun_magnification = 10  # type: float

    @staticmethod
    def equal_area_elevation_angles(n_elevation_quads,  # type: int
                                    max_elevation,  # type: float
                                    max_elevation_units='radians',  # type: str
                                    ):
        if max_elevation_units == 'degrees':
            max_elevation = numpy.deg2rad(max_elevation)
        zenith_min = numpy.pi / 2 - max_elevation
        total_area = (-numpy.cos(numpy.pi / 2.0) + numpy.cos(zenith_min))
        each_area = total_area / n_elevation_quads
        zenith_angles = [zenith_min]
        for quad_num in range(n_elevation_quads):
            theta_n = zenith_angles[-1]
            theta_n_plus_one = numpy.arccos(numpy.cos(theta_n) - each_area)
            zenith_angles.append(theta_n_plus_one)
        elevation_angles = numpy.pi / 2 - numpy.asarray(zenith_angles)
        elevation_angles = numpy.flip(elevation_angles)
        return elevation_angles

    @classmethod
    def create_with_equal_az_el_spacings(cls,
                                         n_azimuths,  # type: int
                                         n_elevations,  # type: int
                                         max_elevation_degrees,  # type: int
                                         ):
        hemisphere = cls()
        hemisphere.azimuth_angles = numpy.linspace(0, 2 * numpy.pi, n_azimuths + 1)
        hemisphere.elevation_angles = numpy.linspace(0, numpy.deg2rad(max_elevation_degrees), n_elevations + 1)
        return hemisphere

    @classmethod
    def create_with_equal_areas(cls,
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

    # TODO: allow polygon to be an ndarray or a shapely polygon
    def az_el_polygon_to_xyz_polygon(self,
                                     polygon,  # type: Polygon
                                     ):
        # TODO: put in an exception if the center_xyz has not been set
        angles = polygon.exterior.coords
        az_el_coords = [a for a in angles]
        xyzs = []
        for az_el in az_el_coords:
            xyz = coordinate_conversions.az_el_r_to_xyz(self._radius, numpy.pi / 2 - az_el[1], az_el[0])
            xyz = numpy.array(xyz) + numpy.array(self.center_xyz)
            xyzs.append(list(xyz))
        poly = Polygon(xyzs)
        return poly

    # TODO: allow polygon to be an ndarray or a shapely polygon
    # TODO: investigate not using shapely at all, their support for 3D polygons isn't great
    def xyz_polygon_to_az_el_polygon(self,
                                     x_arr,  # type: ndarray
                                     y_arr,  # type: ndarray
                                     z_arr,  # type: ndarray
                                     ):
        x_local = x_arr - self.center_xyz[0]
        y_local = y_arr - self.center_xyz[1]
        z_local = z_arr - self.center_xyz[2]

        az, el, r = coordinate_conversions.xyz_to_az_el_radius(x_local, y_local, z_local)

        az_el_poly_coords = []
        for a, e in zip(az, el):
            az_el_poly_coords.append((a, e))
        az_el_poly = Polygon(az_el_poly_coords)
        return az_el_poly

    def burn_az_el_poly_onto_uv_image(self,
                                      az_el_poly,  # type: Polygon
                                      rgb_color,  # type: [int, int, int]
                                      ):
        az_coords, el_coords = az_el_poly.boundary.coords.xy
        az_coords = numpy.array(az_coords)
        el_coords = numpy.array(el_coords)
        pixel_coords = \
            hemisphere_coordinate_conversions.az_el_to_uv_pixel_yx_coords(self.uv_npixels, az_coords, el_coords)
        rr, cc = skimage_polygon(pixel_coords[0], pixel_coords[1])
        self._uv_image[rr, cc, 0] = rgb_color[0]
        self._uv_image[rr, cc, 1] = rgb_color[1]
        self._uv_image[rr, cc, 2] = rgb_color[2]

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

    def all_quad_center_az_els(self, units='radians'):
        azimuth_centers = (self.azimuth_angles[0:-1] + self.azimuth_angles[1:]) / 2
        elevation_centers = (self.elevation_angles[0:-1] + self.elevation_angles[1:]) / 2
        azimuths = numpy.tile(azimuth_centers, self.n_elevation_quads)
        elevations = numpy.repeat(elevation_centers, self.n_azimuth_quads)
        if units.lower() == 'degrees':
            azimuths = numpy.rad2deg(azimuths)
            elevations = numpy.rad2deg(elevations)
        return azimuths, elevations

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
        az_vertices = numpy.tile(self.azimuth_angles[:-1], self.n_elevation_quads + 1)
        el_vertices = numpy.repeat(self.elevation_angles, self.n_azimuth_quads)
        return az_vertices, el_vertices

    @property
    def all_az_el_vertices(self):
        az_vertices, el_vertices = self.quad_az_el_vertices
        return numpy.append(az_vertices, 0), numpy.append(el_vertices, numpy.pi / 2)

    @property
    def quad_xyz_vertices(self):
        az, el = self.quad_az_el_vertices
        xyzs = coordinate_conversions.az_el_r_to_xyz(az, el, combine_arrays=True)
        return xyzs

    @property
    def quad_faces(self):
        quad_faces = []
        quad_face = numpy.array([0, 1, self.n_azimuth_quads + 1, self.n_azimuth_quads])
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
        cap_xyzs = coordinate_conversions.az_el_r_to_xyz(0, numpy.pi / 2, combine_arrays=True)
        # self._az_els_to_xyz(numpy.array([0]), numpy.array([numpy.pi / 2]))
        all_xyz_vertices = numpy.append(quad_xyz_vertices, cap_xyzs, axis=0)
        cap_vertex_index = len(all_xyz_vertices) - 1
        quad_xyz_vertices[-self.n_azimuth_quads:, :]
        top_index_end = len(quad_xyz_vertices)
        top_index_start = len(quad_xyz_vertices) - self.n_azimuth_quads

        tri_faces = trimesh.geometry.triangulate_quads(quad_faces)
        tri_faces = list(tri_faces)

        for i in numpy.arange(top_index_start, top_index_end - 1):
            cap_face = [i, i + 1, cap_vertex_index]
            tri_faces.append(cap_face)
        tri_faces.append([top_index_end - 1, top_index_start, cap_vertex_index])

        hemisphere_trimesh = trimesh.Trimesh()

        hemisphere_trimesh.vertices = all_xyz_vertices
        hemisphere_trimesh.faces = tri_faces

        self._trimesh_model = hemisphere_trimesh

    @property
    def trimesh_model(self):
        return self._trimesh_model

    def color_quad_uv(self,
                      az_index,  # type: int
                      el_index,  # type: int
                      rgb_color,  # type: [int, int, int]
                      ):
        az_el_polygon = self.get_quad_az_els_by_az_el_indices([az_index], [el_index])[0]
        self.burn_az_el_poly_onto_uv_image(az_el_polygon, rgb_color)

    def color_cap_uv(self,
                     rgb_color,  # type: [int, int, int]
                     ):
        cap_polygon = self.cap_polygon_radians
        az_coords, el_coords = cap_polygon.boundary.coords.xy
        az_coords = numpy.array(az_coords)
        el_coords = numpy.array(el_coords)
        pixel_coords = \
            hemisphere_coordinate_conversions.uv_coords_to_uv_pixel_yx_coords(self.uv_npixels,
                                                                              az_coords,
                                                                              el_coords)
        rr, cc = skimage_polygon(pixel_coords[0], pixel_coords[1])
        self._uv_image[rr, cc, 0] = rgb_color[0]
        self._uv_image[rr, cc, 1] = rgb_color[1]
        self._uv_image[rr, cc, 2] = rgb_color[2]

    @property
    def uv_npixels(self):
        ny, nx, bands = self._uv_image.shape
        return nx

    def initialize_uv_image(self,
                            n_pixels=1024,
                            uv_base_color=(0, 0, 255)
                            ):
        # TODO: ensure n_pixels is a power of 2
        self._uv_image = numpy.zeros((n_pixels, n_pixels, 3), dtype=numpy.uint8)
        self._uv_image[:, :, 0] = uv_base_color[0]
        self._uv_image[:, :, 1] = uv_base_color[1]
        self._uv_image[:, :, 2] = uv_base_color[2]

    @property
    def uv_image(self):
        return self._uv_image

    @uv_image.setter
    def uv_image(self,
                 val,  # type: ndarray
                 ):
        self._uv_image = val

    def create_solid_pattern_hemisphere_checkerboard_uv_image(self,
                                                              base_color,  # type: [int, int, int]
                                                              dark_level,  # type: float
                                                              n_pixels=1024,  # type: int
                                                              ):
        # TODO: ensure n_pixels is a power of 2
        self._uv_image = numpy.zeros((n_pixels, n_pixels, 3), dtype=numpy.uint8)
        self._uv_image[:, :, 0] = base_color[0]
        self._uv_image[:, :, 1] = base_color[1]
        self._uv_image[:, :, 2] = base_color[2]

        dark_color = [int(base_color[0] * dark_level),
                      int(base_color[1] * dark_level),
                      int(base_color[2] * dark_level)]

        for el in range(0, self.n_elevation_quads):
            if numpy.mod(el, 2) == 0:
                azimuths = range(0, self.n_azimuth_quads, 2)
            else:
                azimuths = range(1, self.n_elevation_quads, 2)
            for az in azimuths:
                self.color_quad_uv(az, el, dark_color)

    def add_sun_to_uv_image(self, solar_azimuth, solar_elevation, n_sun_polygon_sizes=50):
        disk_elevation_at_vertical = numpy.pi / 2 - numpy.deg2rad(self._sun_size_degrees / 2 * self._sun_magnification)

        disk_elevations_at_vertical = numpy.zeros(n_sun_polygon_sizes + 1) + disk_elevation_at_vertical
        disk_azimuths_at_vertical = numpy.linspace(0, numpy.pi * 2)
        disk_azimuths_at_vertical = numpy.append(disk_azimuths_at_vertical, 0)

        disk_xyzs = coordinate_conversions.az_el_r_to_xyz(disk_azimuths_at_vertical,
                                                          disk_elevations_at_vertical,
                                                          combine_arrays=True)

        solar_zenith = numpy.pi / 2 - solar_elevation

        m_matrix = photogrammetry_utils.create_M_matrix(0, solar_zenith, solar_azimuth, order='yrp')

        rotated_xyzs = disk_xyzs @ m_matrix

        new_az, new_el, new_r = coordinate_conversions.xyz_to_az_el_radius(rotated_xyzs[:, 0],
                                                                           rotated_xyzs[:, 1],
                                                                           rotated_xyzs[:, 2])

        pixel_y_coords, pixel_x_coords = \
            hemisphere_coordinate_conversions.az_el_to_uv_pixel_yx_coords(self.uv_npixels, new_az, new_el)

        rr, cc = skimage_polygon(pixel_y_coords, pixel_x_coords)
        cc[numpy.where(cc >= self.uv_npixels)] = self.uv_npixels - 1
        self._uv_image[rr, cc, 0] = 255
        self._uv_image[rr, cc, 1] = 215
        self._uv_image[rr, cc, 2] = 0

    def apply_uv_image(self):
        texture_image_data = numpy.array(self._uv_image, dtype=numpy.uint8)
        texture_pil_image = Image.fromarray(texture_image_data, 'RGB')

        texture_visual = trimesh.visual.TextureVisuals()
        texture_visual.material.image = texture_pil_image
        self.trimesh_model.visual = texture_visual
        az_vertices, el_vertices = self.all_az_el_vertices
        u_coords, v_coords = hemisphere_coordinate_conversions.az_el_to_uv_coords(az_vertices,
                                                                                  el_vertices)
        uv_coords = numpy.stack((u_coords, v_coords)).transpose()
        texture_visual.uv = uv_coords

    def quad_center_az_els_to_csv(self, output_csv_fname, units='radians'):
        az, el = self.all_quad_center_az_els(units=units)
        az_el_arr = numpy.array((az, el)).transpose()
        numpy.savetxt(output_csv_fname, az_el_arr, delimiter=',', fmt='%1.8f')
