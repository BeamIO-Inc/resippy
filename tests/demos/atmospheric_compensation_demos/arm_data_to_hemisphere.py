"""Examples of using pyrender for viewing and offscreen rendering.
"""
import os
import pyglet
import numpy
import trimesh
import matplotlib.pyplot as plt
from resippy.atmospheric_compensation.hemisphere_quads_model import HemisphereQuadsModel
from resippy.atmospheric_compensation.arm_climate_model import ArmClimateModel
from resippy.utils.image_utils import image_utils

from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     OffscreenRenderer, Mesh, Scene, Node, Viewer

pyglet.options['shadow_window'] = False


#==============================================================================
# Mesh creation
#==============================================================================

texture_visual = trimesh.visual.TextureVisuals()

hemisphere = HemisphereQuadsModel.create_from_equal_areas(50, 50, 89)
hemisphere.center_xyz = [-50, 0, 0]
hemisphere._radius = 0.1
quads = hemisphere.all_quad_xyzs
hemisphere.create_trimesh_model()

arm_png_path = os.path.expanduser("~/Data/SMART/arm_data/sgpirsiviscldmaskC1.a1.20190714.172247.jpg.2019-07-14T17-22-47_RBratio_N2_mask.png")
arm_model = ArmClimateModel.from_image_file(arm_png_path)

# arm_model.project_arm_image_to_cloud_deck(30000)
arm_uv_image = image_utils.resize_image(arm_model._cloud_mask, 1024, 1024)

#hemisphere.create_solid_pattern_hemisphere_checkerboard_uv_image([0, 0, 255], 0.5, n_pixels=2048)
hemisphere.uv_image = arm_uv_image
# hemisphere.color_cap_uv([0, 0, 200])

# add clouds (just boxes for now)
cloud_alt = 100

cloud_1_xs = numpy.array([-20, 20, 20, -20, -20])
cloud_1_ys = numpy.array([-20, -20, 20, 20, -20])
cloud_1_zs = numpy.array([100, 100, 100, 100, 100])

cloud_1_az_el_polygon = hemisphere.xyz_polygon_to_az_el_polygon(cloud_1_xs, cloud_1_ys, cloud_1_zs)
# hemisphere.burn_az_el_poly_onto_uv_image(cloud_1_az_el_polygon, [255, 255, 255])

# hemisphere.add_sun_to_uv_image(numpy.deg2rad(90), numpy.deg2rad(80))


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
direc_l_node = scene.add(direc_l, pose=cam_pose)
spot_l_node = scene.add(spot_l, pose=cam_pose)

#==============================================================================
# Using the viewer with a default camera
#==============================================================================
# v = Viewer(scene, shadows=False)

#==============================================================================
# Using the viewer with a pre-specified camera
#==============================================================================
v = Viewer(scene, central_node=hemisphere_node)

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
