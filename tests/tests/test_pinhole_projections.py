import numpy
import unittest
from resippy.utils import proj_utils
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.supporting_classes.pinhole_camera import PinholeCamera
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.ideal_pinhole_fpa_local_utm_point_calc import IdealPinholeFpaLocalUtmPointCalc
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.fpa_distortion_mapped_point_calc import FpaDistortionMappedPointCalc
from resippy.utils.image_utils import image_utils
from resippy.photogrammetry.lens_distortion_models import BrownConradyDistortionModel
from resippy.utils.units import ureg
from pyproj import transform
from resippy.photogrammetry import crs_defs

center_lon_dd, center_lat_dd, center_alt = -77.025543, 38.889886, 10000
local_projection = proj_utils.decimal_degrees_to_local_utm_proj(center_lon_dd, center_lat_dd)

local_x, local_y = transform(crs_defs.PROJ_4326, local_projection, center_lon_dd, center_lat_dd)

focal_length = 60
focal_length_units = 'mm'
linear_units = 'meters'

omega = 5.230489
phi = 4.98789734
kappa = 23.203784
# omega = 0
# phi = 0
# kappa = 0
angle_units = 'degrees'


class TestCrsTools(unittest.TestCase):
    def test_pixel_to_world_and_back(self):
        pinhole_camera = PinholeCamera()
        pinhole_camera.init_pinhole_from_coeffs(local_x,
                                                local_y,
                                                center_alt,
                                                omega,
                                                phi,
                                                kappa,
                                                focal_length,
                                                x_units=linear_units,
                                                y_units=linear_units,
                                                z_units=linear_units,
                                                omega_units=angle_units,
                                                phi_units=angle_units,
                                                kappa_units=angle_units,
                                                focal_length_units=focal_length_units)

        world_z = 100

        pp = 5*ureg.parse_expression('micrometer').to('meters').magnitude
        fpa_coords_meters = image_utils.create_image_plane_grid(640, 480, pp, pp)

        fpa_coords_meters_x = fpa_coords_meters[0].ravel()
        fpa_coords_meters_y = fpa_coords_meters[1].ravel()
        world_x, world_y = pinhole_camera.image_to_world_plane(fpa_coords_meters_x,
                                                               fpa_coords_meters_y,
                                                               world_z)
        fpa_coords_x_from_world, fpa_coords_y_from_world = pinhole_camera.world_to_image_plane(world_x,
                                                                                               world_y,
                                                                                               world_z)

        assert numpy.isclose(fpa_coords_meters_x, fpa_coords_x_from_world).all()
        assert numpy.isclose(fpa_coords_meters_y, fpa_coords_y_from_world).all()
        print("Successfully projected points from the image plane to the world and back, they match.")

    def test_pixel_to_world_and_back_for_fpa_pinhole_camera(self):
        camera = IdealPinholeFpaLocalUtmPointCalc.init_from_wgs84_params_and_roll_pitch_yaw(center_lon_dd,
                                                                                            center_lat_dd,
                                                                                            center_alt,
                                                                                            omega,
                                                                                            phi,
                                                                                            kappa,
                                                                                            npix_x=640,
                                                                                            npix_y=480,
                                                                                            pixel_pitch_x=5,
                                                                                            pixel_pitch_y=5,
                                                                                            focal_length=focal_length,
                                                                                            alt_units=linear_units,
                                                                                            omega_units=angle_units,
                                                                                            phi_units=angle_units,
                                                                                            kappa_units=angle_units,
                                                                                            pixel_pitch_x_units='micrometer',
                                                                                            pixel_pitch_y_units='micrometer',
                                                                                            focal_length_units=focal_length_units)
        world_z = 100
        pixel_grid = image_utils.create_pixel_grid(camera._npix_x, camera._npix_y)
        lons, lats = camera.pixel_x_y_alt_to_lon_lat(pixel_grid[0], pixel_grid[1], world_z)
        pixels_x, pixels_y = camera.lon_lat_alt_to_pixel_x_y(lons, lats, world_z)

        assert numpy.isclose(pixels_x, pixel_grid[0]).all()
        assert numpy.isclose(pixels_y, pixel_grid[1]).all()
        print("Projections to ground and back match original pixel locations")

    def test_distortion_mapped_fpa_camera(self):
        nx_pixels = 640
        ny_pixels = 480
        pp_meters = 5e-6
        world_z = 100
        distortion_camera = FpaDistortionMappedPointCalc()
        distortion_camera.set_xyz_using_wgs84_coords(center_lon_dd, center_lat_dd, center_alt)
        distortion_camera.set_roll_pitch_yaw(omega, phi, kappa, units=angle_units)
        distortion_camera.set_boresight_roll_pitch_yaw_offsets(0, 0, 0)
        distortion_camera.set_mounting_position_on_fixture(0, 0, 0.0000000)

        dist_model = BrownConradyDistortionModel(0, 0, numpy.asarray([0, 0]), numpy.asarray([0, 0]))
        distorted_grid_x, distorted_grid_y = image_utils.create_image_plane_grid(nx_pixels, ny_pixels, pp_meters, pp_meters)
        undistorted_grid_x, undistorted_grid_y = dist_model.compute_undistorted_image_plane_locations(distorted_grid_x,
                                                                                                      distorted_grid_y)
        distortion_camera.set_distorted_fpa_image_plane_points(undistorted_grid_x, undistorted_grid_y)
        distortion_camera.set_focal_length(focal_length, focal_length_units)

        pixel_grid = image_utils.create_pixel_grid(nx_pixels, ny_pixels)

        lons, lats = distortion_camera._pixel_x_y_alt_to_lon_lat_native(pixel_grid[0], pixel_grid[1], world_z)

        camera = IdealPinholeFpaLocalUtmPointCalc.init_from_wgs84_params_and_roll_pitch_yaw(center_lon_dd,
                                                                                            center_lat_dd,
                                                                                            center_alt,
                                                                                            omega,
                                                                                            phi,
                                                                                            kappa,
                                                                                            npix_x=nx_pixels,
                                                                                            npix_y=ny_pixels,
                                                                                            pixel_pitch_x=pp_meters,
                                                                                            pixel_pitch_y=pp_meters,
                                                                                            focal_length=focal_length,
                                                                                            alt_units=linear_units,
                                                                                            omega_units=angle_units,
                                                                                            phi_units=angle_units,
                                                                                            kappa_units=angle_units,
                                                                                            pixel_pitch_x_units='meters',
                                                                                            pixel_pitch_y_units='meters',
                                                                                            focal_length_units=focal_length_units)
        lons_ideal, lats_ideal = camera.pixel_x_y_alt_to_lon_lat(pixel_grid[0], pixel_grid[1], world_z)

        assert numpy.isclose(lons, lons_ideal).all()
        assert numpy.isclose(lats, lats_ideal).all()

        print("fpa distortion model results with zero modeled distortion matches simple fpa pinhole model going from pixel to ground.")


if __name__ == '__main__':
    unittest.main()