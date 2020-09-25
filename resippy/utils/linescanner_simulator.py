from resippy.image_objects.image_factory import GeotiffImageFactory
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.supporting_classes.fixtured_camera import FixturedCamera
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.ideal_pinhole_fpa_local_utm_point_calc import IdealPinholeFpaLocalUtmPointCalc
from resippy.utils import photogrammetry_utils
from resippy.utils.image_utils import image_utils
from resippy.image_objects.image_factory import ImageFactory
from resippy.image_objects.earth_overhead.physical_camera.physical_camera_metadata import PhysicalCameraMetadata
from pyproj import Proj

import numpy
from scipy.ndimage import map_coordinates


class LinescannerSimulator:
    def __init__(self,
                 geotiff_fname,
                 linescanner_point_calc,
                 ):
        self._geotiff_fname = geotiff_fname
        self._focal_length = focal_length
        self._pixel_pitch_x = pixel_pitch_x
        self._pixel_pitch_y = pixel_pitch_y
        self._npix_x = npix_x
        self._npix_y = npix_y
        self._boresight_roll = boresight_roll
        self._boresight_pitch = boresight_pitch
        self._boresight_yaw = boresight_yaw
        self._pixel_pitch_units = pixel_pitch_units
        self._focal_length_units = focal_length_units
        self._boresight_units = boresight_units
        self._boresight_rpy_order = boresight_rpy_order
        self._external_orientation_spatial_units = external_orientation_spatial_units
        self._external_orientation_rpy_units = external_orientation_rpy_units

        self._gtiff_image_object = GeotiffImageFactory.from_file(geotiff_fname)
        if world_projection is None:
            self._world_projection = self._gtiff_image_object.pointcalc.get_projection()
        else:
            self._world_projection = world_projection

        self._gtiff_image_data = None

        # some cleanup
        if numpy.isclose(self._boresight_roll, 0):
            self._boresight_roll += 1e-8
        if numpy.isclose(self._boresight_pitch, 0):
            self._boresight_pitch += 1e-8
        if numpy.isclose(self._boresight_yaw, 0):
            self._boresight_yaw += 1e-8

    @property
    def world_projection(self):  # type: (...) -> Proj
        return self._world_projection

    @world_projection.setter
    def world_projection(self, val):
        self._world_projection = val

    def _generate_camera_point_calc(self,
                                    lon,
                                    lat,
                                    alt,
                                    roll,
                                    pitch,
                                    yaw,
                                    ):
        fixtured_camera = FixturedCamera()

        fixtured_camera.set_boresight_matrix_from_camera_relative_rpy_params(self._boresight_roll,
                                                                             self._boresight_pitch,
                                                                             self._boresight_yaw,
                                                                             roll_units=self._boresight_units,
                                                                             pitch_units=self._boresight_units,
                                                                             yaw_units=self._boresight_units,
                                                                             order=self._boresight_rpy_order)

        fixtured_camera.set_fixture_orientation_by_roll_pitch_yaw(roll, pitch, yaw,
                                                                  roll_units=self._boresight_units,
                                                                  pitch_units=self._boresight_units,
                                                                  yaw_units=self._boresight_units)

        camera_1_m_matrix = fixtured_camera.get_camera_absolute_M_matrix()
        omega, phi, kappa = photogrammetry_utils.solve_for_omega_phi_kappa(camera_1_m_matrix)

        point_calc = IdealPinholeFpaLocalUtmPointCalc.init_from_local_params(lon,
                                                                             lat,
                                                                             alt,
                                                                             self.world_projection,
                                                                             omega,
                                                                             phi,
                                                                             kappa,
                                                                             self._npix_x,
                                                                             self._npix_y,
                                                                             self._pixel_pitch_x,
                                                                             self._pixel_pitch_y,
                                                                             self._focal_length,
                                                                             alt_units=self._external_orientation_spatial_units,
                                                                             pixel_pitch_x_units=self._pixel_pitch_units,
                                                                             pixel_pitch_y_units=self._pixel_pitch_units,
                                                                             focal_length_units=self._focal_length_units,
                                                                             flip_y=True)
        return point_calc

    def create_overhead_image_object(self,
                                     lon,
                                     lat,
                                     alt,
                                     roll,
                                     pitch,
                                     yaw):
        point_calc = self._generate_camera_point_calc(lon, lat, alt, roll, pitch, yaw)

        pixel_grid = image_utils.create_pixel_grid(self._npix_x, self._npix_y)
        pass1_lons, pass1_lats = point_calc.pixel_x_y_alt_to_lon_lat(pixel_grid[0], pixel_grid[1], pixel_grid[0] * 0)

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
