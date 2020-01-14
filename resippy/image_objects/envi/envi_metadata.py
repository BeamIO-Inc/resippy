from __future__ import division

from resippy.image_objects.abstract_image_metadata import AbstractImageMetadata
import resippy.utils.envi_utils as envi_utils

class EnviMetadata(AbstractImageMetadata):
    def __init__(self):
        super(EnviMetadata, self).__init__()
        envi_header = {}

    @classmethod
    def init_from_file(cls, envi_fname):
        header = envi_utils.read_envi_header(envi_fname)
        envi_metadata = cls()
        envi_metadata.set_npix_x(int(header['samples']))
        envi_metadata.set_npix_y(int(header['lines']))
        envi_metadata.set_n_bands(int(header['bands']))
        envi_metadata.envi_header = header
        return envi_metadata
