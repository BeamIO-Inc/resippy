from numpy import ndarray
import numpy as np


class MMI:
    def __init__(self):
        pass

    @classmethod
    def image_mmi(cls,
                  greyscale_image_1,        # type: ndarray
                  greyscale_image_2,        # type: ndarray
                  nbins=100,                # type: int
                  ):                        # type: (...) -> float

        image1_flat = greyscale_image_1.ravel()
        image2_flat = greyscale_image_2.ravel()
        hist1 = np.histogram(image1_flat, bins=nbins)[0]
        hist2 = np.histogram(image2_flat, bins=nbins)[0]
        joint_histogram = np.histogram2d(image1_flat, image2_flat, bins=nbins)[0]
        mmi = cls._histogram_mmi(hist1, hist2, joint_histogram)
        return mmi

    @classmethod
    def _histogram_mmi(cls,
                       histogram_1,        # type: ndarray
                       histogram_2,        # type: ndarray
                       joint_histogram,    # type: ndarray
                       ):                  # type: (...) -> float
        histogram_1 = histogram_1.astype(float)
        histogram_2 = histogram_2.astype(float)
        joint_histogram = joint_histogram.astype(float)
        n_bins = len(histogram_1)
        mmi = 0
        for n in range(n_bins):
            for m in range(n_bins):
                if joint_histogram[n, m] != 0:
                    tmp = joint_histogram[n, m] * np.log(joint_histogram[n, m] / (histogram_1[n] * histogram_2[m]))
                    mmi += tmp
        return mmi
