from __future__ import division

from resippy.image_objects.earth_overhead.earth_overhead_image_objects.\
    geotiff.geotiff_image_factory import GeotiffImageFactory
from resippy.image_objects.earth_overhead.earth_overhead_image_objects.\
    micasense.micasense_image_factory import MicasenseImageFactory
from resippy.image_objects.envi.envi_image_factory import EnviImageFactory


class ImageFactory:
    def __init__(self):
        pass

    geotiff = GeotiffImageFactory
    micasense = MicasenseImageFactory
    envi = EnviImageFactory
