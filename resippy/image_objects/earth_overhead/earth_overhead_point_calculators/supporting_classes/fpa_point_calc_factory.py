from resippy.photogrammetry.lens_distortion_models.abstract_lens_distortion_model import AbstractLensDistortionModel
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.fpa_distortion_mapped_point_calc import \
    FpaDistortionMappedPointCalc
from resippy.utils.image_utils import image_utils
from resippy.utils.units.unit_constants import ureg


class FPAPointCalcFactory:
    @staticmethod
    def from_distortion_model_and_fpa_loc(distortion_model,  # type: AbstractLensDistortionModel
                                          focal_length,  # type: float
                                          focal_length_units,  # type: str
                                          npix_x,  # type: int
                                          npix_y,  # type: int
                                          pixel_pitch_x,  # type: float
                                          pixel_pitch_y,  # type: float
                                          fpa_x_center=0,  # type: float
                                          fpa_y_center=0,  # type: float
                                          pixel_pitch_units='micrometers',  # type: str
                                          fpa_xy_center_units='micrometers'):  \
            # type: (...) -> FpaDistortionMappedPointCalc
        ppx = pixel_pitch_x * ureg.parse_units(pixel_pitch_units)
        ppy = pixel_pitch_y * ureg.parse_units(pixel_pitch_units)

        image_plane_grid_x, image_plane_grid_y = image_utils.create_image_plane_grid(npix_x,
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
        point_calc = FpaDistortionMappedPointCalc()
        point_calc.set_focal_length(focal_length, focal_length_units)
        point_calc.set_distorted_fpa_image_plane_points(distorted_image_plane_locations[0],
                                                        distorted_image_plane_locations[1])
        return point_calc
