from resippy.image_objects.envi.envi_metadata import EnviMetadata


class MakoMetadata(EnviMetadata):

    def __init__(self):
        super(MakoMetadata, self).__init__()

    def get_gps_timestamp(self):
        return self.envi_header['acquisition time']

    def get_center(self):
        return self.envi_header['geo points']
