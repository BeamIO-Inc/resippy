from __future__ import division

from resippy.image_objects.earth_overhead.digital_globe.view_ready_stereo.view_ready_stereo_image \
    import ViewReadyStereoImage


class ViewReadyStereoImageFactory:
    @staticmethod
    def from_file(fname  # type:  str
                  ):  # type: (...) -> ViewReadyStereoImage
        vrs_image = ViewReadyStereoImage.init_from_file(fname)
        return vrs_image
