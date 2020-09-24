import numpy
import unittest
from resippy.utils import proj_utils
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.pinhole_camera import PinholeCamera
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


class TestCrsTools(unittest.TestCase):
    def test_pixel_to_world_and_back(self):
        omega = 5.230489
        phi = 4.98789734
        kappa = 23.203784
        angle_units = 'degrees'

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


if __name__ == '__main__':
    unittest.main()