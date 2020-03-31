from resippy.image_objects.earth_overhead.abstract_earth_overhead_image import AbstractEarthOverheadImage
from resippy.image_objects.envi.envi_image import EnviImage


class TamariskImage(EnviImage, AbstractEarthOverheadImage):

    def __init__(self):
        super(EnviImage, self).__init__()
        super(AbstractEarthOverheadImage, self).__init__()

    def get_gps_timestamp_and_center(self):
        pass
