from __future__ import division

from resippy.image_objects.envi.envi_image import EnviImage


class EnviImageFactory:
    @staticmethod
    def from_file(fname,                # type:  str
                  header_fname=None     # type: str
                  ):  # type: (...) -> EnviImage
        envi_image = EnviImage.init_from_file(fname, header_fname)
        return envi_image
