import resippy.photogrammetry.crs_defs as crs_defs
from resippy.image_objects.image_factory import ImageFactory
import numpy as np
import os
import json
from resippy.photogrammetry.dem.constant_elevation_dem import ConstantElevationDem
from pyproj import Proj
import matplotlib.pyplot as plt
from resippy.photogrammetry import projection_tools
import time
from resippy.utils.image_utils import geotiff_cropper
from shapely.geometry import Polygon
from resippy.utils.image_utils import image_utils
import imageio


def compute_image_histogram_within_world_polygon(geotiff_fname,  # type: str
                                                  band_num,  # type: int
                                                  world_poly,  # type: np.ndarray
                                                  world_proj,  # type: Proj
                                                  bins=100,
                                                  range=None,
                                                  normed=None,
                                                  weights=None,
                                                  density=None,
                                                 tmp_fname=None,
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

    shapely_poly = Polygon(world_poly)
    if tmp_fname is None:
        tmp_fname = "/tmp/tmp_gtiff.tif"

    geotiff_cropper.crop_geotiff_w_world_polygon(geotiff_fname, tmp_fname, shapely_poly, world_proj)

    image_object = ImageFactory.geotiff.from_file(tmp_fname)
    image_mask = projection_tools.world_poly_to_image_mask(world_poly, image_object, band_num, ConstantElevationDem(), world_proj)
    test_im = image_object.read_band_from_disk(0)
    # get only the pixels within the image mask
    pixels_for_histogram = test_im[np.where(image_mask)]
    hist = np.histogram(pixels_for_histogram, bins=bins, range=range, normed=normed, weights=weights, density=density)
    return hist


def apply_ndvi_colormap(geotiff_fname,  # type: str
                        world_poly,  # type: np.ndarray
                        world_proj,  # type: Proj
                        output_fname,
                        tmp_fname=None,     # type: str
                        min=0,
                        max=1
                        ):
    ndvi_colormap = np.zeros((3, 3))
    ndvi_colormap[0, :] = [255, 32, 0]
    ndvi_colormap[1, :] = [255, 255, 0]
    ndvi_colormap[2, :] = [32, 255, 0]

    shapely_poly = Polygon(world_poly)
    if tmp_fname is None:
        tmp_fname = "/tmp/tmp_gtiff.tif"

    geotiff_cropper.crop_geotiff_w_world_polygon(geotiff_fname, tmp_fname, shapely_poly, world_proj)
    image_object = ImageFactory.geotiff.from_file(tmp_fname)
    # image_object = ImageFactory.geotiff.from_file(geotiff_fname)
    # image_mask = projection_tools.world_poly_to_image_mask(world_poly, image_object, 0, ConstantElevationDem(), world_proj)

    grayscale_image = image_object.read_band_from_disk(0)

    ndvi_image = image_utils.apply_colormap_to_grayscale_image2(grayscale_image, ndvi_colormap, min_cutoff=min, max_cutoff=max)
    imageio.imsave(output_fname, ndvi_image)


def main():
    # set up fnames
    geotiff_fname = os.path.expanduser("~/Data/test_ndvi_stats/NDVI.data.tif")
    geojson_fname = os.path.expanduser("~/Data/test_ndvi_stats/ndvi_test.geojson")

    # get polygons
    with open(geojson_fname) as f:
        features = json.load(f)["features"]
    world_polygon = features[0]['geometry']['coordinates'][0][0]
    # tic = time.time()
    # hist = compute_image_histogram_within_world_polygon(geotiff_fname, 0, world_polygon, crs_defs.PROJ_4326, bins=100)
    # toc = time.time()
    # print("finished computing masked stats in: " + str(toc-tic) + " seconds")

    apply_ndvi_colormap(geotiff_fname, world_polygon, crs_defs.PROJ_4326, os.path.expanduser("~/Downloads/ndvi_image2.png"))


if __name__ == '__main__':
    main()
