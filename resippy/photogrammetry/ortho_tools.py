from __future__ import division

import warnings
from typing import Union

import numpy as np
from numpy.core.multiarray import ndarray
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union
from pyproj import Proj

from resippy.image_objects.earth_overhead.abstract_earth_overhead_image import AbstractEarthOverheadImage
from resippy.image_objects.earth_overhead.earth_overhead_image_objects.geotiff.\
    geotiff_image_factory import GeotiffImageFactory
from resippy.image_objects.earth_overhead.earth_overhead_image_objects.geotiff.\
    geotiff_image import GeotiffImage
from resippy.photogrammetry.dem.abstract_dem import AbstractDem
from resippy.photogrammetry import crs_defs as crs_defs
from resippy.utils.photogrammetry_utils import create_ground_grid, world_poly_to_geo_t
from resippy.photogrammetry.dem.dem_factory import DemFactory
from resippy.utils.image_utils import image_utils as image_utils


# TODO this needs a lot of work, it is very rough and approximate right now
def get_pixel_lon_lats(overhead_image,  # type: AbstractEarthOverheadImage
                       dem=None,  # type: AbstractDem
                       band=None,  # type: list
                       pixels_x=None,  # type: int
                       pixels_y=None,  # type: int
                       pixel_error_threshold=0.01,  # type: float
                       max_iter=1000  # type: int
                       ):  # type: (...) -> (ndarray, ndarray)

    point_calc = overhead_image.get_point_calculator()
    if pixels_x is None or pixels_y is None:
        pixels_x, pixels_y = image_utils.create_pixel_grid(overhead_image.get_metadata().get_npix_x(),
                                                           overhead_image.get_metadata().get_npix_y())
    alts = dem.get_highest_alt()
    lons_highest, lats_highest = point_calc.pixel_x_y_alt_to_lon_lat(pixels_x, pixels_y, alts,
                                                                     band=band,
                                                                     pixel_error_threshold=pixel_error_threshold,
                                                                     max_iter=max_iter)
    # get differences from the highest alt using the dem
    alt_diff = alts - dem.get_elevations(lons_highest, lats_highest)
    new_alts = alts - alt_diff
    lons_low, lats_low = point_calc.pixel_x_y_alt_to_lon_lat(pixels_x, pixels_y, new_alts,
                                                             band=band,
                                                             pixel_error_threshold=pixel_error_threshold,
                                                             max_iter=max_iter)
    return lons_low, lats_low


# TODO more testing on non 4326 projections
def create_ortho_gtiff_image_world_to_sensor(overhead_image,  # type: AbstractEarthOverheadImage
                                             ortho_nx_pix,  # type: int
                                             ortho_ny_pix,  # type: int
                                             world_polygon,  # type: Polygon
                                             world_proj=crs_defs.PROJ_4326,  # type: Proj
                                             dem=None,  # type: AbstractDem
                                             bands=None,  # type: List[int]
                                             nodata_val=0,  # type: float
                                             output_fname=None,  # type: str
                                             interpolation='nearest'  # type: str
                                             ):  # type:  (...) -> GeotiffImage

    envelope = world_polygon.envelope
    minx, miny, maxx, maxy = envelope.bounds
    image_ground_grid_x, image_ground_grid_y = create_ground_grid(minx, maxx, miny, maxy, ortho_nx_pix, ortho_ny_pix)
    geo_t = world_poly_to_geo_t(envelope, ortho_nx_pix, ortho_ny_pix)

    if dem is None:
        dem = DemFactory.constant_elevation(0)
        dem.set_projection(crs_defs.PROJ_4326)
    alts = dem.get_elevations(image_ground_grid_x, image_ground_grid_y, world_proj)

    if bands is None:
        bands = list(range(overhead_image.get_metadata().get_n_bands()))

    images = []
    for i, band in enumerate(bands):
        pixels_x, pixels_y = overhead_image.get_point_calculator(). \
            lon_lat_alt_to_pixel_x_y(image_ground_grid_x, image_ground_grid_y, alts, band=i, world_proj=world_proj)
        image_data = overhead_image.read_band_from_disk(band)
        im_tp = image_data.dtype

        regridded = image_utils.grid_warp_image_band(image_data, pixels_x, pixels_y,
                                                     nodata_val=nodata_val, interpolation=interpolation)
        regridded = regridded.astype(im_tp)
        images.append(regridded)

    orthorectified_image = np.stack(images, axis=2)
    gtiff_image = GeotiffImageFactory.from_numpy_array(orthorectified_image, geo_t, world_proj)
    gtiff_image.get_metadata().set_nodata_val(nodata_val)
    if output_fname is not None:
        gtiff_image.write_to_disk(output_fname)

    return gtiff_image


def get_extent(overhead_image,  # type: AbstractEarthOverheadImage
               dem=None,  # type: AbstractDem
               bands=None,  # type: Union[int, list]
               pixel_error_threshold=0.01,  # type: float
               max_iter=20  # type: int
               ):  # type: (...) -> Polygon
    """
    :param overhead_image
    :param dem:
    :param bands:
    :param pixel_error_threshold:
    :param max_iter:
    :return:

    This will return image extents in the CRS specified by the DEM.  If no DEM is provided, the extents will be native
    To the image object.
    """

    nx = overhead_image.get_metadata().get_npix_x()
    ny = overhead_image.get_metadata().get_npix_y()
    x = np.arange(0, nx)
    y = np.arange(0, ny)
    xx, yy = np.meshgrid(x, y, sparse=False)
    x_border = np.concatenate((xx[0, :], xx[:, -1], xx[-1, :], xx[:, 0]))
    y_border = np.concatenate((yy[0, :], yy[:, -1], yy[-1, :], yy[:, 0]))

    if type(bands) == (type(1)):
        bands = [bands]

    if bands is None and not overhead_image.get_point_calculator().bands_coregistered():
        bands = np.arange(overhead_image.get_metadata().get_n_bands())
        warnings.warn("the bands for this image aren't co-registered.  Getting the full image extent using all bands.")

    all_band_lons = np.array([])
    all_band_lats = np.array([])
    for band in bands:
        lons, lats = get_pixel_lon_lats(overhead_image, pixels_x=x_border, pixels_y=y_border, dem=dem,
                                        pixel_error_threshold=pixel_error_threshold,
                                        max_iter=max_iter,
                                        band=band)
        all_band_lons = np.append(all_band_lons, lons)
        all_band_lats = np.append(all_band_lats, lats)

    extents = [(MultiPoint(list(zip(all_band_lons, all_band_lats))))]
    # TODO, check into whether we can get rid of this tolerance, also, this isn't a very good approach, since it assumes
    # TODO a particular CRS (probably 4326)
    return unary_union(extents).convex_hull.simplify(0.001)
