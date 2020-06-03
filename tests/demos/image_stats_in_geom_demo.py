import resippy.utils.pix4d_utils as pix4d
import resippy.utils.photogrammetry_utils as photogrammetry_utils
import resippy.photogrammetry.crs_defs as crs_defs
import resippy.photogrammetry.ortho_tools as ortho_tools
from resippy.photogrammetry.dem.dem_factory import DemFactory
from resippy.image_objects.image_factory import ImageFactory
from tests import demo_data_base_dir, demo_data_save_dir
from resippy.utils import file_utils as file_utils
import numpy as np
import os
import json
from shapely.geometry import shape, GeometryCollection
from resippy.image_objects.earth_overhead.abstract_earth_overhead_image import AbstractEarthOverheadImage
from resippy.photogrammetry.dem.abstract_dem import AbstractDem
from pyproj import Proj
from shapely.geometry import Polygon
import shapely
from resippy.utils import numpy_and_array_utils
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
from resippy.utils.image_utils import image_utils


micasense_dir = file_utils.get_path_from_subdirs(demo_data_base_dir, ['image_data',
                                                                      'multispectral',
                                                                      'micasense',
                                                                      '20181019_hana',
                                                                      '1703',
                                                                      'micasense',
                                                                      'processed'
                                                                      ])

geoson_file = file_utils.get_path_from_subdirs(demo_data_base_dir, ['geo_data',
                                                                    'micasense_20181019_hana',
                                                                    'micasense_multiple_polygons.geojson']
                                               )


band_fname_dict = {}
for cam_num in range(1, 6):
    band_num = str(cam_num)
    band_fname_dict['band' + band_num] = \
        file_utils.get_path_from_subdirs(micasense_dir, ['merged', 'L0_IMG_0404_' + band_num + '.tif'])

# set up path to pix4d parameters used to build point calculator etc
params_dir = file_utils.get_path_from_subdirs(micasense_dir, ['pix4d_1703_mica_imgs85_1607',
                                                              '1_initial',
                                                              'params'])

# set up where to save output orthos
save_dir = os.path.join(demo_data_save_dir, "micasense_ortho_demos")
file_utils.make_dir_if_not_exists(save_dir)

# set up where digital surface model is stored
dsm_fullpath = file_utils.get_path_from_subdirs(micasense_dir,
                                                ['pix4d_1703_mica_imgs85_1607',
                                                 '3_dsm_ortho',
                                                 'pix4d_1704_mica_85_1607_dsm.tif'])

# load up the pix4d info
pix4d_master_dict = pix4d.make_master_dict(params_dir)

# make the micasense camera object
micasense_image = ImageFactory.micasense.from_image_number_and_pix4d(band_fname_dict, pix4d_master_dict)
micasense_native_projection = micasense_image.get_point_calculator().get_projection()

# make the dem camera object
dem_image = DemFactory.from_gtiff_file(dsm_fullpath)
dem_native_projection = dem_image.get_projection()

if dem_native_projection.srs != micasense_native_projection.srs:
    raise ValueError("pix4d dem and micasense projections should be in same CRS!")


elevation = np.nanmean(dem_image.dem_data)
const_elevation_dem = DemFactory.constant_elevation(elevation)


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


def combine_image_masks(image_masks,        # type: [np.ndarray]
                        ):
    combined_mask = image_masks[0]
    if len(image_masks) == 1:
        pass
    else:
        for mask in image_masks[1:]:
            combined_mask = np.bitwise_or(combined_mask, mask)
    return combined_mask


def compute_image_histogram_within_world_polygons(image,            # type: AbstractEarthOverheadImage
                                                  band_num,         # type: int
                                                  world_polygons,      # type: []
                                                  world_proj,       # type: Proj
                                                  dem=None,         # type: AbstractDem
                                                  ):
    image_masks = []

    for world_poly in world_polygons:
        image_poly = world_poly_to_image_poly(world_poly, micasense_image, band_num, dem, world_proj)
        image_mask = world_poly_to_image_mask(world_poly, micasense_image, band_num, dem, world_proj)
        image_masks.append(image_mask)

    combined_mask = combine_image_masks(image_masks)

    test_im = micasense_image.read_band_from_disk(0)
    blended = image_utils.blend_images(test_im, combined_mask)
    # get only the pixels within the image mask
    pixels_for_histogram = test_im[np.where(combined_mask)]
    hist = np.histogram(pixels_for_histogram, bins=100)
    stop = 1


def main():
    # get polygons
    world_polygons = []
    with open(geoson_file) as f:
        features = json.load(f)["features"]
    for feature in features:
        world_polygons.append(feature['geometry']['coordinates'][0][0])
    compute_image_histogram_within_world_polygons(micasense_image)


if __name__ == '__main__':
    main()

