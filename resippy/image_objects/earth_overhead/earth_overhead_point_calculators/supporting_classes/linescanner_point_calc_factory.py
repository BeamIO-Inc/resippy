from resippy.photogrammetry.lens_distortion_models.abstract_lens_distortion_model import AbstractLensDistortionModel
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.line_scanner_point_calc import \
    LineScannerPointCalc
from resippy.utils.image_utils import image_utils
from resippy.utils.units.unit_constants import ureg


class LinescannerPointCalcFactory:
    @staticmethod
    def from_distortion_model_and_fpa_loc(distortion_model,  # type: AbstractLensDistortionModel
                                          focal_length,  # type: float
                                          focal_length_units,  # type: str
                                          npix_y,  # type: int
                                          pixel_pitch_x,  # type: float
                                          pixel_pitch_y,  # type: float
                                          fpa_x_center=0,  # type: float
                                          fpa_y_center=0,  # type: float
                                          pixel_pitch_units='micrometers',  # type: str
                                          fpa_xy_center_units='micrometers'):  \
            # type: (...) -> LineScannerPointCalc
        ppx = pixel_pitch_x * ureg.parse_units(pixel_pitch_units)
        ppy = pixel_pitch_y * ureg.parse_units(pixel_pitch_units)

        image_plane_grid_x, image_plane_grid_y = image_utils.create_image_plane_grid(1,
                                                                                     npix_y,
                                                                                     ppx,
                                                                                     ppy)
        image_plane_grid_x = image_plane_grid_x + fpa_x_center * ureg.parse_units(fpa_xy_center_units)
        image_plane_grid_y = image_plane_grid_y + fpa_y_center * ureg.parse_units(fpa_xy_center_units)

        image_plane_grid_x = image_plane_grid_x.magnitude
        image_plane_grid_y = image_plane_grid_y.magnitude

        distorted_image_plane_locations = distortion_model.compute_distorted_image_plane_locations(image_plane_grid_x,
                                                                                                   image_plane_grid_y,
                                                                                                   pixel_pitch_units)
        point_calc = LineScannerPointCalc()
        point_calc.set_camera_model_w_distorted_image_plane_grid(distorted_image_plane_locations[0],
                                                                 distorted_image_plane_locations[1],
                                                                 focal_length,
                                                                 focal_length_units)
        return point_calc
