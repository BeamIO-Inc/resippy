from __future__ import division

from resippy.image_objects.abstract_image_metadata import AbstractImageMetadata
import resippy.utils.envi_utils as envi_utils

class EnviMetadata(AbstractImageMetadata):
    def __init__(self):
        super(EnviMetadata, self).__init__()
        envi_header = {}

    @classmethod
    def init_from_header(cls, header_fname):
        header = envi_utils.read_envi_header(header_fname)
        envi_metadata = cls()
        envi_metadata.set_npix_x(header['samples'])
        envi_metadata.set_npix_y(header['lines'])
        envi_metadata.set_n_bands(header['bands'])
        envi_metadata.envi_header = header
        return envi_metadata
