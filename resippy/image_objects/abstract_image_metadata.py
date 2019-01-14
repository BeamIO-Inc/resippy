from __future__ import division

import abc
from six import add_metaclass


@add_metaclass(abc.ABCMeta)
class AbstractImageMetadata():
    """Concrete implementations should initialize an image for reading/writing
    and should also set the image's metadata object and point
    calculator object"""

    def __init__(self):
        self.metadata_dict = {}
        self.set_nodata_val(None)

    def get_npix_x(self):   # type: (...) -> int
        return self.metadata_dict['npix_x']

    def get_npix_y(self):   # type: (...) -> int
        return self.metadata_dict['npix_y']

    def get_n_bands(self):   # type: (...) -> int
        return self.metadata_dict['n_bands']

    def set_npix_x(self,
                   npix_x   # type: int
                   ):       # type: (...) -> None
        self.metadata_dict['npix_x'] = npix_x

    def set_npix_y(self,
                   npix_y   # type: int
                   ):       # type: (...) -> None
        self.metadata_dict['npix_y'] = npix_y

    def set_n_bands(self,
                    n_bands     # type: int
                    ):          # type: (...) -> None
        self.metadata_dict['n_bands'] = n_bands

    def set_nodata_val(self,
                       nodata_val   # type: int
                       ):           # type: (...) -> None
        self.metadata_dict['nodata_val'] = nodata_val

    def get_nodata_val(self):   # type: (...) -> int
        return self.metadata_dict['nodata_val']
