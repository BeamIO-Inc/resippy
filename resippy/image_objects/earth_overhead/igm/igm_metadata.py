from __future__ import division

from resippy.image_objects.abstract_image_metadata import AbstractImageMetadata


class IgmMetadata(AbstractImageMetadata):
    @classmethod
    def from_params(cls,
                    n_x_pixels,   # type: int
                    n_y_pixels,  # type: int
                    n_bands,  # type: int
                    ):  # type: (...) -> IgmMetadata
        igm_metadata = cls()
        igm_metadata.set_npix_x(n_x_pixels)
        igm_metadata.set_npix_y(n_y_pixels)
        igm_metadata.set_n_bands(n_bands)
        return igm_metadata
