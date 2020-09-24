import numpy
import unittest
from resippy.utils import proj_utils
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.pinhole_camera import PinholeCamera
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.ideal_pinhole_fpa_local_utm_point_calc import IdealPinholeFpaLocalUtmPointCalc
from resippy.utils.image_utils import image_utils
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
        camera = IdealPinholeFpaLocalUtmPointCalc.init_from_wgs84_params(center_lon_dd,
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


if __name__ == '__main__':
    unittest.main()