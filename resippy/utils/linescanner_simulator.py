from resippy.image_objects.image_factory import GeotiffImageFactory
from resippy.utils.image_utils import image_utils
from resippy.image_objects.image_factory import ImageFactory
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.line_scanner_point_calc import LineScannerPointCalc
from resippy.image_objects.earth_overhead.physical_camera.physical_camera_metadata import PhysicalCameraMetadata

import numpy
from scipy.ndimage import map_coordinates


class LinescannerSimulator:
    def __init__(self,
                 geotiff_fname,
                 linescanner_point_calc,    # type: LineScannerPointCalc
                 ):
        self._geotiff_fname = geotiff_fname
        self._gtiff_image_object = GeotiffImageFactory.from_file(geotiff_fname)
        self._gtiff_image_data = None
        self._point_calc = linescanner_point_calc  # type: LineScannerPointCalc

    def create_overhead_image_object(self):
        pixel_grid_x, pixel_grid_y = image_utils.create_pixel_grid(1, self._point_calc.n_cross_track_pixels)
        pass1_lons, pass1_lats = self._point_calc.pixel_x_y_alt_to_lon_lat(pixel_grid[0], pixel_grid[1], pixel_grid[0] * 0)

        gtiff_x_vals, gtiff_y_vals = self._gtiff_image_object.get_point_calculator().lon_lat_alt_to_pixel_x_y(
            pass1_lons,
            pass1_lats,
            numpy.zeros_like(pass1_lons))

        if self._gtiff_image_data is None:
            self._gtiff_image_data = self._gtiff_image_object.read_band_from_disk(0)

        simulated_image_band = map_coordinates(self._gtiff_image_data,
                                               [image_utils.flatten_image_band(gtiff_y_vals),
                                                image_utils.flatten_image_band(gtiff_x_vals)])

        simulated_image_band = image_utils.unflatten_image_band(simulated_image_band, self._npix_x, self._npix_y)
        simulated_image_data = numpy.reshape(simulated_image_band, (self._npix_y, self._npix_x, 1))
        metadata = PhysicalCameraMetadata()
        metadata.set_npix_x(self._npix_x)
        metadata.set_npix_y(self._npix_y)
        metadata.set_n_bands(1)

        simulated_image_obj = ImageFactory.physical_camera.from_numpy_array_metadata_and_single_point_calc(simulated_image_data, metadata, point_calc)
        return simulated_image_obj
