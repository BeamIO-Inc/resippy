from __future__ import division

import abc
from six import add_metaclass
from typing import Union


@add_metaclass(abc.ABCMeta)
class AbstractImageMetadata():
    """
    AbstractImageMetadata class.  This class contains metadata about an image Object.   Concrete implementations
    of this class should be bundled with their corresponding Image Object.
    """

    def __init__(self):
        """
        Default initialization method.  This sets the metadata class variables.  Metadata is generally tracked
        using a python dictionary, which is initialized here.  Imagery typically also has
        a "nodata_value", which is set to "None" in the metadata dictionary by default.
        """
        self.metadata_dict = {}
        self.set_nodata_val(None)

    def get_image_name(self):   # type: (...) -> str
        """
        get's the image name.  This is typically the base filename for file-based images, with the path information
        removed.
        :return: image name of type (str)
        """
        return self.metadata_dict['img_name']

    def get_npix_x(self):   # type: (...) -> int
        """
        gets the number of "x" pixels, sometimes referred to as "columns" or "samples"
        :return: number of x pixels
        """
        return self.metadata_dict['npix_x']

    def get_npix_y(self):   # type: (...) -> int
        """
        gets the number of "y" pixels, sometimes referred to as "rows" or "lines"
        :return: number of y pixels
        """
        return self.metadata_dict['npix_y']

    def get_n_bands(self):   # type: (...) -> int
        """
        get number of spectral bands
        :return: number of spectral bands
        """
        return self.metadata_dict['n_bands']

    def set_image_name(self,
                       name  # type: str
                       ):  # type: (...) -> None
        """
        Set or overwrite the image name.  This should be used when creating a new metadata object.
        :param name: string of the image name to set
        :return: None
        """
        self.metadata_dict['img_name'] = name

    def set_npix_x(self,
                   npix_x   # type: int
                   ):       # type: (...) -> None
        """
        sets the number of "x" pixels, sometimes referred to as "columns" or "samples".
        This should be used when creating a new metadata object.
        :param npix_x:
        :return: None
        """
        self.metadata_dict['npix_x'] = npix_x

    def set_npix_y(self,
                   npix_y   # type: int
                   ):       # type: (...) -> None
        """
        sets the number of "y" pixels, sometimes referred to as "rows" or "lines".
        This should be used when creating a new metadata object
        :param npix_y:
        :return: None
        """
        self.metadata_dict['npix_y'] = npix_y

    def set_n_bands(self,
                    n_bands     # type: int
                    ):          # type: (...) -> None
        """
        sets the number of spectral bands.  This should be used when creating a new metadata object
        :param n_bands:
        :return:
        """
        self.metadata_dict['n_bands'] = n_bands

    def set_nodata_val(self,
                       nodata_val   # type: Union[int, float]
                       ):           # type: (...) -> None
        """
        sets the nodata value for a corresponding image.  This value can be used by image processing routines that
        Take an image object as an input and do things like set the nodata value to a transparent layer.
        :param nodata_val: image's nodata value
        :return: None
        """
        self.metadata_dict['nodata_val'] = nodata_val

    def get_nodata_val(self):   # type: (...) -> Union[int, float]
        """
        gets the nodata value for a corresponding image.  This value can be used by image processing routines that
        Take an image object as an input and do things like set the nodata value to a transparent layer.
        :return: nodata value as either an integer or float, depending on the data type of the image
        """
        return self.metadata_dict['nodata_val']
