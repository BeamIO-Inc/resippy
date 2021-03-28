import trimesh
import pyrender
import numpy
import threading
import time

fuze_trimesh = trimesh.load('/Users/jasoncasey/Downloads/pyrender-master/examples/models/fuze.obj')
mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
scene = pyrender.Scene()
scene.add(mesh)

v = pyrender.Viewer(scene, run_in_thread=True, use_raymond_lighting=True)

def event_loop():
  global v

  i = 0
  while v.is_active:
    pose = numpy.eye(4)
    pose[:3,3] = [i, 0, 0]
    v.render_lock.acquire()
    for n in v.scene.mesh_nodes:
      v.scene.set_pose(n, pose)
    v.render_lock.release()
    i += 0.01
    time.sleep(0.1)

t = threading.Thread(target=event_loop)
t.start()