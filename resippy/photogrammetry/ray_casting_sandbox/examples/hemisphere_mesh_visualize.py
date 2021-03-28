"""Examples of using pyrender for viewing and offscreen rendering.
"""
import pyglet
import numpy
import trimesh
import matplotlib.pyplot as plt
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

hemisphere = HemisphereQuadsModel.create_from_equal_areas(30, 30, 80)
hemisphere.center_xyz = [0, 0, 0]
hemisphere._radius = 0.1
quads = hemisphere.all_quad_xyzs
hemisphere.create_trimesh_model()

hemisphere.initialize_uv_image()
hemisphere.color_quad_uv(0, 0, [255, 0, 0])
hemisphere.color_quad_uv(0, 1, [0, 255, 0])
hemisphere.color_quad_uv(0, 2, [255, 255, 0])
hemisphere.color_quad_uv(0, 3, [255, 0, 255])
hemisphere.color_quad_uv(0, 4, [0, 255, 255])
hemisphere.color_quad_uv(0, 5, [255, 0, 0])
hemisphere.color_quad_uv(0, 6, [0, 255, 0])
hemisphere.color_quad_uv(0, 7, [255, 255, 0])
hemisphere.color_quad_uv(0, 8, [255, 0, 255])
hemisphere.color_quad_uv(0, 9, [0, 255, 255])

hemisphere.color_cap_uv([255, 255, 255])

hemisphere.add_sun_to_uv_image(numpy.deg2rad(10), numpy.deg2rad(180))


hemisphere.apply_uv_image()
hemisphere_mesh = Mesh.from_trimesh(hemisphere.trimesh_model)


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
# v = Viewer(scene, shadows=False)

#==============================================================================
# Using the viewer with a pre-specified camera
#==============================================================================
# v = Viewer(scene, central_node=hemisphere_node)
v = Viewer(scene, run_in_thread=False)

# cam_node = scene.add(cam, pose=cam_pose)

# #==============================================================================
# # Rendering offscreen from that camera
# #==============================================================================
#
# r = OffscreenRenderer(viewport_width=640*2, viewport_height=480*2)
# color, depth = r.render(scene)
#
# plt.figure()
# plt.imshow(color)
# plt.show()
#
# r.delete()
