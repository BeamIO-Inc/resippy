import resippy.photogrammetry.crs_defs as crs_defs
from resippy.image_objects.image_factory import ImageFactory
import numpy as np
import os
import json
from resippy.image_objects.earth_overhead.abstract_earth_overhead_image import AbstractEarthOverheadImage
from resippy.photogrammetry.dem.abstract_dem import AbstractDem
from resippy.photogrammetry.dem.constant_elevation_dem import ConstantElevationDem
from pyproj import Proj
import matplotlib.pyplot as plt
from resippy.photogrammetry import projection_tools
from resippy.utils.image_utils import image_mask_utils
import time


def compute_image_histogram_within_world_polygons(image_object,  # type: AbstractEarthOverheadImage
                                                  band_num,  # type: int
                                                  world_polygons,  # type: []
                                                  world_proj,  # type: Proj
                                                  dem=None,  # type: AbstractDem
                                                  bins=100,
                                                  range=None,
                                                  normed=None,
                                                  weights=None,
                                                  density=None,
                                                  ):
    """

    :param image_object: AbstractEarthOverheadImage
    :param band_num: int
    :param world_polygons: list
    :param world_proj: Proj
    :param dem: AbstractDem
    :param bins: int
    :param range:
    :param normed:
    :param weights:
    :param density:
    :return:

    last set of parameters from bins through density correspond to numpy's "histogram" routine to allow all paramters
    to be passed through.
    """

    if dem is None:
        dem = ConstantElevationDem()

    image_masks = []
    for world_poly in world_polygons:
        image_mask = projection_tools.world_poly_to_image_mask(world_poly, image_object, band_num, dem, world_proj)
        image_masks.append(image_mask)

    combined_mask = image_mask_utils.combine_image_masks(image_masks)

    test_im = image_object.read_band_from_disk(0)
    # get only the pixels within the image mask
    pixels_for_histogram = test_im[np.where(combined_mask)]
    hist = np.histogram(pixels_for_histogram, bins=bins, range=range, normed=normed, weights=weights, density=density)
    return hist


def main():
    # set up fnames
    geotiff_fname = os.path.expanduser("~/Data/test_ndvi_stats/NDVI.data.tif")
    geojson_fname = os.path.expanduser("~/Data/test_ndvi_stats/ndvi_test.geojson")

    # get the image object
    ndvi_image = ImageFactory.geotiff.from_file(geotiff_fname)

    # get polygons
    world_polygons = []
    with open(geojson_fname) as f:
        features = json.load(f)["features"]
    for feature in features:
        world_polygons.append(feature['geometry']['coordinates'][0][0])
    tic = time.time()
    hist = compute_image_histogram_within_world_polygons(ndvi_image, 0, world_polygons, crs_defs.PROJ_4326, bins=100)
    toc = time.time()
    print("finished computing masked stats in: " + str(toc-tic) + " seconds")
    plt.plot(hist[0])
    plt.show()


if __name__ == '__main__':
    main()
