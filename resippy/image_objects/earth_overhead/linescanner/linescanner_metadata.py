from __future__ import division

from resippy.image_objects.abstract_image_metadata import AbstractImageMetadata


class LinescannerMetadata(AbstractImageMetadata):

    def set_gdal_datatype(self,
                          dtype  # type: int
                          ):  # type: (...) -> None
        self.metadata_dict["gdal_datatype"] = dtype

    def get_gdal_datatype(self):  # type: (...) -> int
        return self.metadata_dict["gdal_datatype"]
