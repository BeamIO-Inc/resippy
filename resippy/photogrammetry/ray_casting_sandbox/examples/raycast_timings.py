import numpy
import time
from resippy.photogrammetry.ray_casting_sandbox.raycaster import RayCaster


def ray_plane_intersect_timings(nx_pixels, ny_pixels):
    ray_starts = numpy.zeros((nx_pixels*ny_pixels, 3))
    ray_starts[:, 2] = 100
    ray_ends = numpy.zeros_like(ray_starts)
    ray_ends[:, 0] = numpy.linspace(-0.2, 0.2, nx_pixels*ny_pixels)
    ray_ends[:, 1] = numpy.linspace(-0.3, 0.3, nx_pixels*ny_pixels)
    ray_ends[:, 2] = 99

    raycaster = RayCaster()
    z_heights = numpy.linspace(0, 10, nx_pixels*ny_pixels)
    y_locs = numpy.linspace(10, 11, nx_pixels*ny_pixels)
    x_locs = numpy.linspace(10, 11, nx_pixels*ny_pixels)
    raycaster.ray_starts = ray_starts
    raycaster.ray_ends = ray_ends
    tic = time.time()
    xy_points = raycaster.rays_xy_planes_intersections(z_heights)
    xz_points = raycaster.rays_xz_planes_intersections(y_locs)
    yz_points = raycaster.rays_yz_planes_intersections(x_locs)
    toc = time.time()
    print(str(toc-tic) + " seconds to process " + str(nx_pixels*ny_pixels*3) + " rays")


def ray_intersects_box_timings(nx_pixels, ny_pixels):
    ray_starts = numpy.zeros((nx_pixels*ny_pixels, 3))
    ray_starts[:, 2] = 100
    ray_ends = numpy.zeros_like(ray_starts)
    ray_ends[:, 0] = numpy.linspace(-0.2, 0.2, nx_pixels*ny_pixels)
    ray_ends[:, 1] = numpy.linspace(-0.3, 0.3, nx_pixels*ny_pixels)
    ray_ends[:, 2] = 99

    raycaster = RayCaster()
    raycaster.ray_starts = ray_starts
    raycaster.ray_ends = ray_ends
    box_x_min = -1
    box_x_max = 1
    box_y_min = -1
    box_y_max = 1
    box_z_min = 0
    box_z_max = 80
    tic = time.time()
    inersects = raycaster.rays_intersect_boxes(box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max)
    toc = time.time()
    print(str(toc-tic) + " seconds to process " + str(nx_pixels*ny_pixels*3) + " rays")


def main():
    ray_plane_intersect_timings(1000, 1000)
    ray_intersects_box_timings(1000, 1000)


if __name__ == '__main__':
    main()
