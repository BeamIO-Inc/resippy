import numpy
import time
from resippy.image_objects.earth_overhead.geotiff.geotiff_image_factory import GeotiffImageFactory
from resippy.photogrammetry import crs_defs
from resippy.utils import photogrammetry_utils
from shapely.geometry import Point
from resippy.photogrammetry.ray_casting_sandbox import geotiff_dem_to_trimesh
from resippy.utils import proj_utils
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.ideal_pinhole_fpa_local_utm_point_calc import IdealPinholeFpaLocalUtmPointCalc


def create_mesh(geotiff_nx,  # type: int
                geotiff_ny,  # type: int
                extent_x_meters,  # type: int
                extent_y_meters,  # type: int
                ):
    center_lat = 38.8894
    center_lon = -77.0443

    lon_lat_point = Point(center_lon, center_lat)
    local_proj = proj_utils.decimal_degrees_to_local_utm_proj(center_lon, center_lat)
    local_lon_lat = photogrammetry_utils.reproject_geometry(lon_lat_point, crs_defs.PROJ_4326, local_proj)

    extent_per_triangle_x = (extent_x_meters/(geotiff_nx-1))
    extent_per_triangle_y = (extent_y_meters/(geotiff_ny-1))

    x_start = local_lon_lat.x - extent_per_triangle_x*geotiff_nx/2
    y_start = local_lon_lat.y + extent_per_triangle_y*geotiff_ny/2

    geot = [x_start, extent_per_triangle_x, 0, y_start, 0, -extent_per_triangle_y]

    image_data = numpy.zeros((geotiff_ny, geotiff_nx))
    gtiff_image = GeotiffImageFactory.from_numpy_array(image_data, geot, local_proj)
    mesh = geotiff_dem_to_trimesh.geotiff_dem_to_trimesh(gtiff_image)
    return mesh


def create_rays(nx_pixels, ny_pixels, mesh_length_x, mesh_length_y):
    stop = 1

mesh_nx = 100
mesh_ny = 100

pixels_nx = 100
pixels_ny = 100

mesh = create_mesh(mesh_nx, mesh_ny, 1000, 1000)
rays = create_rays()
