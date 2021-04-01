"""Examples of using pyrender for viewing and offscreen rendering.
"""
import pyglet
import numpy
import trimesh
from resippy.photogrammetry.dem.trimesh_dem import TrimeshDem

from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     OffscreenRenderer, Mesh, Scene, Node, Viewer

pyglet.options['shadow_window'] = False


#==============================================================================
# Mesh creation
#==============================================================================

texture_visual = trimesh.visual.TextureVisuals()


def create_ramped_trimesh_dem(nx,  # type: int
                              ny,  # type: int
                              start_elevation,  # type: float
                              end_elevation,  # type: float
                              axis_direction='x',  # type: str
                              ):
    numpy_dem = numpy.zeros((ny, nx))
    if axis_direction == 'x':
        elevations = numpy.linspace(start_elevation, end_elevation, nx)
        for i in range(nx):
            numpy_dem[:, i] = elevations[i]
    else:
        elevations = numpy.linspace(start_elevation, end_elevation, ny)
        for i in range(ny):
            numpy_dem[i, :] = elevations[i]
    dem_geot = (0, 1, 0, ny-1, 0, -1)
    t = TrimeshDem()
    trimesh_dem = t.from_dem_numpy_array(numpy_dem, dem_geot)
    return trimesh_dem


dem = create_ramped_trimesh_dem(50, 100, 0, 10)
dem_mesh = Mesh.from_trimesh(dem.trimesh_model)


#==============================================================================
# Light creation
#==============================================================================

direc_l = DirectionalLight(color=numpy.ones(3),
                           intensity=1.0)
spot_l = SpotLight(color=numpy.ones(3),
                   intensity=1.0,
                   innerConeAngle=numpy.pi/16,
                   outerConeAngle=numpy.pi/6)
point_l = PointLight(color=numpy.ones(3),
                     intensity=10.0)

#==============================================================================
# Camera creation
#==============================================================================

cam = PerspectiveCamera(yfov=(numpy.pi / 3.0))
cam_pose = numpy.array([
    [0.0,  -numpy.sqrt(2)/2, numpy.sqrt(2)/2, 0.5],
    [1.0, 0.0,           0.0,           0.0],
    [0.0,  numpy.sqrt(2)/2,  numpy.sqrt(2)/2, 0.4],
    [0.0,  0.0,           0.0,          1.0]
])

#==============================================================================
# Scene creation
#==============================================================================

scene = Scene(ambient_light=numpy.array([0.7, 0.7, 0.7, 100.0]))
dem_node = Node(mesh=dem_mesh, translation=numpy.array([0.0, 0.0, 0]))
scene.add_node(dem_node)
direc_l_node = scene.add(direc_l, pose=cam_pose)
spot_l_node = scene.add(spot_l, pose=cam_pose)
v = Viewer(scene, central_node=dem_node)
