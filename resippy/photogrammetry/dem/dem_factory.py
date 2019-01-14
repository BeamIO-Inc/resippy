from __future__ import division

from resippy.image_objects.earth_overhead.earth_overhead_image_objects.geotiff.\
    geotiff_image_factory import GeotiffImageFactory
from resippy.photogrammetry.dem.geotiff_dem import GeotiffDem
from resippy.photogrammetry.dem.constant_elevation_dem import ConstantElevationDem


class DemFactory:
    @staticmethod
    def from_gtiff_file(fname,  # type: str
                        nodata_value=None,  # type: float
                        interpolation_method='bilinear', # type: str
                        ):  # type (...) -> GeotiffDem
        # type: (str) -> GeotiffDem
        gtiff = GeotiffImageFactory.from_file(fname)
        gtiff_dem = GeotiffDem()
        gtiff_dem.set_geotiff_image(gtiff)
        # attempt to get nodata value from the gtiff file itself
        if nodata_value is None:
            nodata_value = gtiff.get_metadata().get_nodata_val()
        gtiff_dem.remove_nodata_values(nodata_value)
        if interpolation_method == 'bilinear':
            gtiff_dem.set_interpolation_to_bilinear()
        elif interpolation_method == 'nearest':
            gtiff_dem.set_interpolation_to_nearest()
        else:
            TypeError("interpolation method should either be 'bilinear' or 'nearest'")
        return gtiff_dem

    @staticmethod
    def constant_elevation(
            elevation=0  # type: float
    ):  # type: (...) -> ConstantElevationDem
        return ConstantElevationDem(elevation=elevation)
