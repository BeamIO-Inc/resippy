"""Examples of using pyrender for viewing and offscreen rendering.
"""
import pyglet
import numpy
import numpy as np
import trimesh
from PIL import Image

from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     OffscreenRenderer, Mesh, Scene, Node

pyglet.options['shadow_window'] = False


#==============================================================================
# Mesh creation
#==============================================================================

#------------------------------------------------------------------------------
# Creating textured meshes from trimeshes
#------------------------------------------------------------------------------

# Fuze trimesh
fuze_trimesh = trimesh.load('/Users/jasoncasey/Downloads/pyrender-master/examples/models/fuze.obj')
fuze_texture_path = '/Users/jasoncasey/Downloads/pyrender-master/examples/models/fuze_uv.jpg'
fuze_mesh = Mesh.from_trimesh(fuze_trimesh)

# Wood trimesh
wood_trimesh = trimesh.load('/Users/jasoncasey/Downloads/pyrender-master/examples/models/wood.obj')
wood_mesh = Mesh.from_trimesh(wood_trimesh)

fuze2_trimesh = trimesh.Trimesh()
fuze2_trimesh.vertices = fuze_trimesh.vertices
fuze2_trimesh.faces = fuze_trimesh.faces

texture_pil_image_jpg = Image.open(fuze_texture_path)
texture_image_data = numpy.array(texture_pil_image_jpg)
texture_pil_image = Image.fromarray(texture_image_data, 'RGB')

texture_visual = trimesh.visual.TextureVisuals()
texture_visual.material.image = texture_pil_image
fuze2_trimesh.visual = texture_visual
texture_visual.uv = fuze_trimesh.visual.uv

fuze2_mesh = Mesh.from_trimesh(fuze2_trimesh)


#==============================================================================
# Light creation
#==============================================================================

direc_l = DirectionalLight(color=np.ones(3), intensity=10.0)
spot_l = SpotLight(color=np.ones(3), intensity=10.0,
                   innerConeAngle=np.pi/16, outerConeAngle=np.pi/6)
point_l = PointLight(color=np.ones(3), intensity=10.0)

#==============================================================================
# Camera creation
#==============================================================================

cam = PerspectiveCamera(yfov=(np.pi / 3.0))
cam_pose = np.array([
    [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 0.5],
    [1.0, 0.0,           0.0,           0.0],
    [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 0.4],
    [0.0,  0.0,           0.0,          1.0]
])

#==============================================================================
# Scene creation
#==============================================================================

scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))

#------------------------------------------------------------------------------
# By manually creating nodes
#------------------------------------------------------------------------------
fuze2_node = Node(mesh=fuze2_mesh, translation=np.array([0.1, 0.15, -np.min(fuze_trimesh.vertices[:, 2])]))
scene.add_node(fuze2_node)

#------------------------------------------------------------------------------
# By using the add() utility function
#------------------------------------------------------------------------------
wood_node = scene.add(wood_mesh)
direc_l_node = scene.add(direc_l, pose=cam_pose)
spot_l_node = scene.add(spot_l, pose=cam_pose)

#==============================================================================
# Using the viewer with a default camera
#==============================================================================

# v = Viewer(scene, shadows=True)

#==============================================================================
# Using the viewer with a pre-specified camera
#==============================================================================
cam_node = scene.add(cam, pose=cam_pose)
# v = Viewer(scene, central_node=fuze2_node)

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
