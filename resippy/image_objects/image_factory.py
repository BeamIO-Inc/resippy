from __future__ import division

from resippy.image_objects.earth_overhead.geotiff.geotiff_image_factory import GeotiffImageFactory
from resippy.image_objects.earth_overhead.micasense.micasense_image_factory import MicasenseImageFactory
from resippy.image_objects.envi.envi_image_factory import EnviImageFactory
from resippy.image_objects.earth_overhead.digital_globe.DigitalGlobeImageFactory import DigitalGlobeImageFactory


class ImageFactory:
    """
    This is a factory class that provides convenience routines to create various types of image objects.
    This is basically a class that accumulates other image factory classes and keeps them all in one place,
    So that any type of image object can be instantiated using a single Image Factory.
    Each time a user creates a new type of image object and image factory, that particular image factory
    can be imported in this file and then set as a variable in the main code for the class.

    An image factory should be constructed as a class.  Within that class there should be one or more static methods
    that take input parameters and return a the particular type of image object corresponding to the factory object.
    For instance, and GeotiffImageFactory would have static methods that return GeotiffImage objects.
    """
    def __init__(self):
        pass

    geotiff = GeotiffImageFactory
    micasense = MicasenseImageFactory
    envi = EnviImageFactory
    digital_globe = DigitalGlobeImageFactory
