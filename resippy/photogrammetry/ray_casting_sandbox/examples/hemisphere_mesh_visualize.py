"""Examples of using pyrender for viewing and offscreen rendering.
"""
import pyglet
import numpy
import trimesh
from resippy.atmospheric_compensation.hemisphere_quads_model import HemisphereQuadsModel

from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     OffscreenRenderer, Mesh, Scene, Node, Viewer

pyglet.options['shadow_window'] = False


#==============================================================================
# Mesh creation
#==============================================================================

#------------------------------------------------------------------------------
# Creating textured meshes from trimeshes
#------------------------------------------------------------------------------

# Wood trimesh
wood_trimesh = trimesh.load('/Users/jasoncasey/Downloads/pyrender-master/examples/models/wood.obj')
wood_mesh = Mesh.from_trimesh(wood_trimesh)

texture_visual = trimesh.visual.TextureVisuals()

hemisphere = HemisphereQuadsModel.create_from_equal_az_el_spacings(10, 10, 60)
hemisphere.center_xyz = [0, 0, 0]
hemisphere._radius = 0.1
quads = hemisphere.all_quad_xyzs
hemisphere = hemisphere.create_trimesh_model()

hemisphere_mesh = Mesh.from_trimesh(hemisphere)


#==============================================================================
# Light creation
#==============================================================================

direc_l = DirectionalLight(color=numpy.ones(3), intensity=1.0)
spot_l = SpotLight(color=numpy.ones(3), intensity=1.0,
                   innerConeAngle=numpy.pi/16, outerConeAngle=numpy.pi/6)
point_l = PointLight(color=numpy.ones(3), intensity=10.0)

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

#------------------------------------------------------------------------------
# By manually creating nodes
#------------------------------------------------------------------------------
hemisphere_node = Node(mesh=hemisphere_mesh, translation=numpy.array([0.0, 0.0, 0]))
scene.add_node(hemisphere_node)

#------------------------------------------------------------------------------
# By using the add() utility function
#------------------------------------------------------------------------------
wood_node = scene.add(wood_mesh)
direc_l_node = scene.add(direc_l, pose=cam_pose)
spot_l_node = scene.add(spot_l, pose=cam_pose)

#==============================================================================
# Using the viewer with a default camera
#==============================================================================
v = Viewer(scene, shadows=True)

#==============================================================================
# Using the viewer with a pre-specified camera
#==============================================================================
cam_node = scene.add(cam, pose=cam_pose)
v = Viewer(scene, central_node=hemisphere_node)

#==============================================================================
# Rendering offscreen from that camera
#==============================================================================

r = OffscreenRenderer(viewport_width=640*2, viewport_height=480*2)
color, depth = r.render(scene)

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(color)
plt.show()

#==============================================================================
# Segmask rendering
#==============================================================================

# nm = {node: 20*(i + 1) for i, node in enumerate(scene.mesh_nodes)}
# seg = r.render(scene, RenderFlags.SEG, nm)[0]
# plt.figure()
# plt.imshow(seg)
# plt.show()

r.delete()
