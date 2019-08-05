from __future__ import division

from resippy.image_objects.abstract_image_metadata import AbstractImageMetadata


class MicasenseMetadata(AbstractImageMetadata):
    def __init__(self):
        super(MicasenseMetadata, self).__init__()
        self.set_n_bands(5)
        self.set_npix_x(1280)
        self.set_npix_y(960)
