from osgeo import gdal
from pyproj import Proj
from shapely.geometry import Polygon
from resippy.image_objects.earth_overhead.geotiff.geotiff_image_factory import GeotiffImageFactory
from resippy.utils import photogrammetry_utils


def crop_geotiff_w_world_polygon(input_fname,           # type: str
                                 output_fname,          # type: str
                                 world_poly,            # type: Polygon
                                 world_proj=None,       # type: Proj
                                 ):
    input_image_object = GeotiffImageFactory.from_file(input_fname)
    input_proj = input_image_object.get_point_calculator().get_projection()
    if world_proj is not None:
        if world_proj == input_proj:
            pass
        else:
            world_proj = photogrammetry_utils.reproject_geometry(world_poly, world_proj, input_proj)

    minx, miny, maxx, maxy = world_proj.bounds

    ds = gdal.Open(input_fname)
    ds = gdal.Translate(output_fname, ds, projWin=[minx, maxy, maxx, miny])
    ds = None