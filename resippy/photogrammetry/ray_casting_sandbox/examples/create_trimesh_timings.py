import numpy
import time
from resippy.image_objects.earth_overhead.geotiff.geotiff_image_factory import GeotiffImageFactory
from resippy.photogrammetry.ray_casting_sandbox import geotiff_dem_to_trimesh


def create_mesh(nx,  # type: int
                ny,  # type: int
                ):
    image_data = numpy.arange(0, nx*ny)
    image_data = numpy.reshape(image_data, (ny, nx))
    geot = [0, 1, 0, ny, 0, -1]
    gtiff_image = GeotiffImageFactory.from_numpy_array(image_data, geot, None)
    mesh = geotiff_dem_to_trimesh.geotiff_dem_to_trimesh(gtiff_image)
    stop = 1


nx = 10
ny = 10
tic = time.time()
create_mesh(nx, ny)
toc = time.time()

print(str(nx) + " x " + str(ny) + " took " + str(toc-tic) + " seconds")
