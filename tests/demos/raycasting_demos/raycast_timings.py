import time
import numpy
from resippy.photogrammetry.dem.trimesh_dem import TrimeshDem
import trimesh


def create_constant_elevation_trimesh_dem(nx,  # type: int
                                          ny,  # type: int
                                          elevation,  # type: float
                                          ):
    numpy_dem = numpy.zeros((ny, nx)) + elevation
    dem_geot = (0, 1, 0, ny-1, 0, -1)
    trimesh_dem = TrimeshDem.from_dem_numpy_array(numpy_dem, dem_geot)
    return trimesh_dem


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
    trimesh_dem = TrimeshDem.from_dem_numpy_array(numpy_dem, dem_geot)
    return trimesh_dem


def create_rays(n_rays,  # type: int
                fov_degrees,  # type: float
                location_xyz,  # type: (float, float, float)
                ):
    ray_origins = numpy.zeros((n_rays, 3))
    ray_origins[:, 0] = location_xyz[0]
    ray_origins[:, 1] = location_xyz[1]
    ray_origins[:, 2] = location_xyz[2]
    fov_radians = numpy.deg2rad(fov_degrees)
    ray_directions = numpy.zeros_like(ray_origins)
    ray_directions[:, 0] = numpy.linspace(-fov_radians, fov_radians, n_rays)
    ray_directions[:, 1] = 0
    ray_directions[:, 2] = -1
    return ray_origins, ray_directions


def run_dem_ray_timings(dem_nx,
                        dem_ny,
                        n_rays,
                        ):
    dem_mesh = create_ramped_trimesh_dem(dem_nx, dem_ny, 0, 10)
    ray_starts, ray_ends = create_rays(n_rays, 10, (dem_nx / 2, dem_ny / 2, 100))
    tic = time.time()
    locations, index_ray, index_tri = dem_mesh.trimesh_model.ray.intersects_location(ray_starts, ray_ends)
    toc = time.time()
    print("took " + str(toc - tic) + " seconds to cast " + str(n_rays) +
          " rays onto a " + str(dem_nx) + "x" + str(dem_ny) + "dem.")
    print(str(locations.shape[0]) + " rays out of " + str(n_rays) + " hit a surface.")


def dem_100x100_n_rays_1000_timings():
    run_dem_ray_timings(1000, 200, 100000)


def main():
    dem_100x100_n_rays_1000_timings()
    # dem_100x100_n_rays_1000_pyembree_timings()


if __name__ == '__main__':
    main()
