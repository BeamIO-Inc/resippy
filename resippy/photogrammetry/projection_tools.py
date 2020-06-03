import numpy as np
from resippy.image_objects.earth_overhead.abstract_earth_overhead_image import AbstractEarthOverheadImage
from resippy.photogrammetry.dem.abstract_dem import AbstractDem
from pyproj import Proj
from resippy.utils import numpy_and_array_utils
from PIL import Image
from PIL import ImageDraw


# world poly on x/y coords (lon/lat), image poly in y/x coords
def world_poly_to_image_poly(world_polygon,             # type: []
                             image_object,              # type: AbstractEarthOverheadImage
                             band_num,                  # type: int
                             dem=None,                  # type: AbstractDem
                             world_proj=None,           # type: Proj
                             ):
    lons = np.array(world_polygon)[:, 0]
    lats = np.array(world_polygon)[:, 1]
    image_poly_coords_x, image_poly_coords_y = image_object.get_point_calculator().lon_lat_to_pixel_x_y(lons, lats, dem, world_proj, band_num)
    image_poly = numpy_and_array_utils.separate_lists_to_list_of_tuples([image_poly_coords_x, image_poly_coords_y])
    return image_poly


def world_poly_to_image_mask(world_polygon,             # type: []
                             image_object,              # type: AbstractEarthOverheadImage
                             band_num,                  # type: int
                             dem=None,                  # type: AbstractDem
                             world_proj=None,           # type: Proj
                             ):
    image_poly = world_poly_to_image_poly(world_polygon, image_object, band_num, dem, world_proj)
    image_mask = Image.new('1',
                           (image_object.get_metadata().get_npix_x(), image_object.get_metadata().get_npix_y()),
                           0)
    ImageDraw.Draw(image_mask).polygon(image_poly, outline=1, fill=1)
    mask = np.array(image_mask)
    return mask