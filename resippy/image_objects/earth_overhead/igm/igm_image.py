from typing import Union
from numpy import ndarray
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.igm_point_calc import IGMPointCalc
from resippy.image_objects.earth_overhead.abstract_earth_overhead_image import AbstractEarthOverheadImage
from resippy.image_objects.earth_overhead.igm.igm_metadata import IgmMetadata
from pyproj import Proj


class IgmImage(AbstractEarthOverheadImage):
    @classmethod
    def from_params(cls,
                    image_data,  # type: ndarray
                    lons,  # type: ndarray
                    lats,  # type: ndarray
                    alts,  # type: Union[ndarray, float]
                    n_bands,  # type: int
                    projection,  # type: Proj
                    ):
        igm_image = cls()
        igm_image._image_data = image_data
        igm_image._point_calc = IGMPointCalc(lons, lats, alts, projection)
        n_y, n_x = lons.shape
        metadata = IgmMetadata.from_params(n_x, n_y, n_bands)
        igm_image.set_metadata(metadata)
        igm_image.point_calc.set_projection(projection)
        return igm_image

    def read_band_from_disk(self, band_number):
        pass

    def read_all_image_data_from_disk(self):
        pass

    @property
    def point_calc(self):
        return self._point_calc
