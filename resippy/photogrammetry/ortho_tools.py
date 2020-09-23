from __future__ import division

import warnings
from typing import Union

import numpy as np
import math
from numpy.core.multiarray import ndarray
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union
from pyproj import Proj
from scipy.ndimage import map_coordinates

from resippy.image_objects.earth_overhead.abstract_earth_overhead_image import AbstractEarthOverheadImage
from resippy.image_objects.earth_overhead.geotiff.geotiff_image_factory import GeotiffImageFactory
from resippy.image_objects.earth_overhead.geotiff.geotiff_image import GeotiffImage
from resippy.image_objects.earth_overhead.igm.igm_image import IgmImage
from resippy.photogrammetry.dem.abstract_dem import AbstractDem
from resippy.photogrammetry import crs_defs as crs_defs
from resippy.utils.photogrammetry_utils import create_ground_grid, world_poly_to_geo_t
from resippy.utils import proj_utils
from pyproj import transform
from resippy.photogrammetry.dem.dem_factory import DemFactory
from resippy.utils.image_utils import image_utils as image_utils

import matplotlib.pyplot as plt


def get_pixel_values(image_object,  # type: AbstractEarthOverheadImage
                     lons,  # type: ndarray
                     lats,  # type: ndarray
                     alts,  # type: ndarray
                     window=8  # type: int
                     ):  # type: (...) -> ndarray

    nodata = image_object.get_metadata().get_nodata_val()
    wh = np.round(window/2.0).astype(np.int)
    pix_vals = []
    for band in range(image_object.get_metadata().get_n_bands()):
        img = image_object.read_band_from_disk(band)
        x, y = image_object.get_point_calculator().lon_lat_alt_to_pixel_x_y(lons, lats, alts, band=band)
        locs = zip(np.round(x).astype(np.int), np.round(y).astype(np.int))
        vals = []
        for item in locs:
            try:
                xmin = item[0] - wh
                xmax = item[0] + wh
                ymin = item[1] - wh
                ymax = item[1] + wh
                pix_win = img[ymin:ymax, xmin:xmax]
                good_inds = pix_win != nodata
                val = np.mean(pix_win[good_inds])
            except IndexError:
                val = np.NaN
            vals.append(val)
        pix_vals.append(vals)

    return np.array(pix_vals)


# TODO replace this with the ray caster once it is tested and more robust.
def get_pixel_lon_lats(overhead_image,  # type: AbstractEarthOverheadImage
                       dem=None,  # type: AbstractDem
                       band=None,  # type: list
                       pixels_x=None,  # type: int
                       pixels_y=None,  # type: int
                       pixel_error_threshold=0.01,  # type: float
                       max_iter=1000  # type: int
                       ):  # type: (...) -> (ndarray, ndarray)

    if dem is None:
        dem = DemFactory.constant_elevation(0)
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


def create_igm_image(overhead_image,    # type: AbstractEarthOverheadImage
                     dem=None,          # type: AbstractDem
                     dem_sample_distance=None,  # type: int
                     ):  # type: (...) -> IgmImage
    if dem is None:
        dem = DemFactory.constant_elevation()
    pixels_x, pixels_y = image_utils.create_pixel_grid(overhead_image.metadata.get_npix_x(),
                                                       overhead_image.metadata.get_npix_y())
    lons, lats, alts = overhead_image.pointcalc.pixel_x_y_to_lon_lat_alt(pixels_x,
                                                                         pixels_y,
                                                                         dem,
                                                                         dem_sample_distance=dem_sample_distance)
    igm_image = IgmImage.from_params(overhead_image.get_image_data(),
                                     lons,
                                     lats,
                                     alts,
                                     overhead_image.get_image_data().shape[2],
                                     overhead_image.pointcalc.get_projection())
    return igm_image


def create_full_ortho_gtiff_image(overhead_image,  # type: AbstractEarthOverheadImage
                                  gsd_meters=None,  # type: float
                                  dem=None,  # type: AbstractDem
                                  output_proj=None,  # type: Proj
                                  bands=None,  # type: [int]
                                  nodata_val=0,  # type: float
                                  spline_order=1,  # type: str
                                  ):  # type: (...) -> GeotiffImage
    extent = get_extent(overhead_image, dem, bands=bands)
    native_lon, native_lat = extent.bounds[0], extent.bounds[1]
    lon_4326, lat_4326 = transform(overhead_image.pointcalc.get_projection(), crs_defs.PROJ_4326, native_lon, native_lat)
    local_proj = proj_utils.decimal_degrees_to_local_utm_proj(lon_4326, lat_4326)
    if gsd_meters is None:
        nx = overhead_image.metadata.get_npix_x()
        ny = overhead_image.metadata.get_npix_y()
        native_lons, native_lats = get_pixel_lon_lats(overhead_image,
                                                      dem,
                                                      bands,
                                                      pixels_x=[0, nx-1, 0],
                                                      pixels_y=[0, 0, ny-1])
        local_lons, local_lats = transform(overhead_image.pointcalc.get_projection(), local_proj, native_lons, native_lats)
        distance_along_x = math.sqrt((local_lons[1] - local_lons[0]) ** 2 + (local_lats[1] - local_lats[0]) ** 2)
        distance_along_y = math.sqrt((local_lons[2] - local_lons[0]) ** 2 + (local_lats[2] - local_lats[0]) ** 2)
        gsd_x = distance_along_x / nx
        gsd_y = distance_along_y / ny
        gsd_meters = (gsd_x + gsd_y) / 2.0

    ortho_x_dist = np.abs(extent.bounds[2] - extent.bounds[0])
    ortho_y_dist = np.abs(extent.bounds[3] - extent.bounds[1])

    ortho_nx = int(ortho_x_dist / gsd_meters)
    ortho_ny = int(ortho_y_dist / gsd_meters)

    if output_proj is None:
        output_proj = overhead_image.pointcalc.get_projection()
    return create_ortho_gtiff_image_world_to_sensor(overhead_image,
                                                    ortho_nx,
                                                    ortho_ny,
                                                    extent,
                                                    world_proj=output_proj,
                                                    dem=dem,
                                                    bands=bands,
                                                    nodata_val=nodata_val,
                                                    spline_order=spline_order)

def mask_image(image, nodata_val=0):
    """
    Mask tiff images to make the nodata_val region transparent. One alpha channel is added to the image.
    Value of 0 for alpha: full transparent
    Value of 255 for alpha: fully opaque

    # Note: Given that by default black pixels in channel 0 / band 1 are masked, masking gets broken
    for relatively darker images.

    :param image: a single or multi-band (orthorectified) image.
    :return: input image with an added alpha layer which masks nodata_val region.
    """
    layer = image[:, :, 0]  # Take band 0 as the reference layer to mask the image.
    alpha = layer.copy()
    alpha[layer == nodata_val] = 0
    alpha[layer != nodata_val] = 255
    image = np.dstack((image, alpha))
    return image

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
                                             spline_order=0,  # type: str
                                             mask_no_data_region=False,  # type: bool
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
    if overhead_image.get_point_calculator().bands_coregistered() is not True:
        for band in bands:
            pixels_x, pixels_y = overhead_image.get_point_calculator(). \
                lon_lat_alt_to_pixel_x_y(image_ground_grid_x, image_ground_grid_y, alts, band=band, world_proj=world_proj)
            image_data = overhead_image.read_band_from_disk(band)
            im_tp = image_data.dtype

            regridded = map_coordinates(image_data,
                                        [image_utils.flatten_image_band(pixels_y),
                                         image_utils.flatten_image_band(pixels_x)],
                                        order=spline_order)

            regridded = image_utils.unflatten_image_band(regridded, ortho_nx_pix, ortho_ny_pix)
            regridded = regridded.astype(im_tp)
            images.append(regridded)
    else:
        pixels_x, pixels_y = overhead_image.get_point_calculator(). \
            lon_lat_alt_to_pixel_x_y(image_ground_grid_x, image_ground_grid_y, alts, band=0, world_proj=world_proj)
        for band in bands:
            image_data = overhead_image.get_image_band(band)
            if image_data is None:
                image_data = overhead_image.read_band_from_disk(band)
            im_tp = image_data.dtype

            regridded = map_coordinates(image_data,
                                        [image_utils.flatten_image_band(pixels_y),
                                         image_utils.flatten_image_band(pixels_x)],
                                        order=spline_order)

            regridded = image_utils.unflatten_image_band(regridded, ortho_nx_pix, ortho_ny_pix)

            regridded = regridded.astype(im_tp)

            regridded[np.where(pixels_x <= 0)] = nodata_val
            regridded[np.where(pixels_x >= overhead_image.metadata.get_npix_x() - 2)] = nodata_val

            regridded[np.where(pixels_y <= 0)] = nodata_val
            regridded[np.where(pixels_y >= overhead_image.metadata.get_npix_y() - 2)] = nodata_val
            images.append(regridded)

    orthorectified_image = np.stack(images, axis=2)

    if mask_no_data_region:
        orthorectified_image = mask_image(orthorectified_image, nodata_val)

    gtiff_image = GeotiffImageFactory.from_numpy_array(orthorectified_image, geo_t, world_proj)
    gtiff_image.get_metadata().set_nodata_val(nodata_val)
    if output_fname is not None:
        gtiff_image.write_to_disk(output_fname)

    return gtiff_image


def create_ortho_gtiff_image_world_to_sensor_backup(overhead_image,  # type: AbstractEarthOverheadImage
                                             ortho_nx_pix,  # type: int
                                             ortho_ny_pix,  # type: int
                                             world_polygon,  # type: Polygon
                                             world_proj=crs_defs.PROJ_4326,  # type: Proj
                                             dem=None,  # type: AbstractDem
                                             bands=None,  # type: List[int]
                                             nodata_val=0,  # type: float
                                             output_fname=None,  # type: str
                                             interpolation='nearest',  # type: str
                                             mask_no_data_region=False, # type: bool
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
    if overhead_image.get_point_calculator().bands_coregistered() is not True:
        for band in bands:
            pixels_x, pixels_y = overhead_image.get_point_calculator(). \
                lon_lat_alt_to_pixel_x_y(image_ground_grid_x, image_ground_grid_y, alts, band=band, world_proj=world_proj)
            image_data = overhead_image.read_band_from_disk(band)
            im_tp = image_data.dtype

            regridded = image_utils.grid_warp_image_band(image_data, pixels_x, pixels_y,
                                                         nodata_val=nodata_val, interpolation=interpolation)
            regridded = regridded.astype(im_tp)
            images.append(regridded)
    else:
        pixels_x, pixels_y = overhead_image.get_point_calculator(). \
            lon_lat_alt_to_pixel_x_y(image_ground_grid_x, image_ground_grid_y, alts, band=0, world_proj=world_proj)
        for band in bands:
            image_data = overhead_image.get_image_band(band)
            if image_data is None:
                image_data = overhead_image.read_band_from_disk(band)
            im_tp = image_data.dtype
            regridded = image_utils.grid_warp_image_band(image_data, pixels_x, pixels_y,
                                                         nodata_val=nodata_val, interpolation=interpolation)
            regridded = regridded.astype(im_tp)
            images.append(regridded)

    orthorectified_image = np.stack(images, axis=2)
    if mask_no_data_region:
        orthorectified_image = mask_image(orthorectified_image, nodata_val)

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

    if bands is None:
        bands = np.arange(overhead_image.get_metadata().get_n_bands())

    all_band_lons = np.array([])
    all_band_lats = np.array([])

    if not overhead_image.get_point_calculator().bands_coregistered():
        warnings.warn("the bands for this image aren't co-registered.  Getting the full image extent using all bands.")
        for band in bands:
            lons, lats = get_pixel_lon_lats(overhead_image, pixels_x=x_border, pixels_y=y_border, dem=dem,
                                            pixel_error_threshold=pixel_error_threshold,
                                            max_iter=max_iter,
                                            band=band)
            all_band_lons = np.append(all_band_lons, lons)
            all_band_lats = np.append(all_band_lats, lats)
    else:
        lons, lats = get_pixel_lon_lats(overhead_image, pixels_x=x_border, pixels_y=y_border, dem=dem,
                                        pixel_error_threshold=pixel_error_threshold,
                                        max_iter=max_iter,
                                        band=0)
        all_band_lons = np.append(all_band_lons, lons)
        all_band_lats = np.append(all_band_lats, lats)

    extents = [(MultiPoint(list(zip(all_band_lons, all_band_lats))))]
    # TODO, check into whether we can get rid of this tolerance, also, this isn't a very good approach, since it assumes
    # TODO a particular CRS (probably 4326)
    return unary_union(extents).convex_hull.simplify(0.001)


def are_pixels_obstructed_by_dem(earth_overhead_image,      # type: AbstractEarthOverheadImage
                                 pixels_x,                  # type: ndarray
                                 pixels_y,              # type: ndarray
                                 lons,  # type: ndarray
                                 lats,  # type: ndarray
                                 dem,   # type: AbstractDem
                                 alts=None,     # type: ndarray
                                 dem_resolution=None,   # type: float
                                 band=0,                # type: int
                                 ):  # type: (...) -> ndarray
    DEFAULT_DEM_RESOLUTION = 5
    if dem_resolution is None:
        dem_resolution = DEFAULT_DEM_RESOLUTION
    highest_alts = np.zeros_like(pixels_x)
    highest_alts[:] = dem.get_highest_alt() + 0.001

    if alts is None:
        alts = dem.get_elevations(lons, lats)

    obstructed_mask = np.zeros_like(lons)

    lons_highest, lats_highest = earth_overhead_image.get_point_calculator().pixel_x_y_alt_to_lon_lat(pixels_x, pixels_y, highest_alts, band=band)
    lon_diffs = lons - lons_highest
    lat_diffs = lats - lats_highest
    horizontal_distances = np.sqrt(np.square(lon_diffs + lat_diffs))
    vertical_distances = highest_alts - alts
    max_horizontal_distance = horizontal_distances.max()
    n_dem_steps = max_horizontal_distance / dem_resolution
    for step in np.arange(1, n_dem_steps):
        test_lons = lons + lon_diffs * (step) / n_dem_steps
        test_lats = lats + lat_diffs * (step) / n_dem_steps
        test_alts = dem.get_elevations(test_lons, test_lats)
        obstruction_heights = alts + vertical_distances * step / n_dem_steps
        obstructed_mask[np.where(test_alts > obstruction_heights)] = 1
    return obstructed_mask
