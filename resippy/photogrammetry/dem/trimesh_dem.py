import numpy
from numpy import ndarray
import trimesh


class TrimeshDem:
    def __init__(self):
        self._trimesh_model = None  # type: trimesh.Trimesh

    @classmethod
    def from_dem_numpy_array(cls,
                             numpy_array,  # type: ndarray
                             geot,  # type: (float, float, float, float, float, float)
                             ):
        ny, nx = numpy_array.shape
        x_tmp = numpy.arange(0, nx) * geot[1] + geot[0]
        y_tmp = numpy.arange(0, ny) * geot[-1] + geot[3]
        x_coords = numpy.tile(x_tmp, ny)
        y_coords = numpy.repeat(y_tmp, nx)
        vertices = numpy.zeros((ny*nx, 3))
        vertices[:, 0] = x_coords
        vertices[:, 1] = y_coords
        vertices[:, 2] = numpy_array.flatten()
        quad_faces_ul = numpy.repeat(numpy.arange(0, ny-1), nx-1) * (nx) + numpy.tile(numpy.arange(0, nx-1), ny-1)
        quad_faces_ur = quad_faces_ul + 1
        quad_faces_ll = quad_faces_ul + nx
        quad_faces_lr = quad_faces_ll + 1
        quad_faces = numpy.array([quad_faces_ul, quad_faces_ur, quad_faces_ll, quad_faces_lr]).transpose()
        tri_faces = trimesh.geometry.triangulate_quads(quad_faces)
        trimesh_model = trimesh.Trimesh(vertices, tri_faces)
        trimesh_dem = cls()
        trimesh_dem._trimesh_model = trimesh_model
        return trimesh_dem

    @property
    def trimesh_model(self):
        return self._trimesh_model

