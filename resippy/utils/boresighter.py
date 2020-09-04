import cv2 as cv
import matplotlib.pyplot as plt

from resippy.image_objects.earth_overhead.physical_camera.physical_camera_image import PhysicalCameraImage
from resippy.photogrammetry.dem.abstract_dem import AbstractDem
from resippy.photogrammetry import ortho_tools
from resippy.utils.image_utils import image_utils
import numpy


class SiftBoresighter:
    def __init__(self,
                 image_objects,  # type: List[AbstractEarthOverheadImage]
                 dem=None,          # type: AbstractDem
                 ):
        self._image_objects = image_objects  # type: AbstractEarthOverheadImage
        self._dem = dem

    def get_image_object(self,
                         index,  # type: int
                         ):     # type: (...) -> AbstractEarthOverheadImage
        return self._image_objects[index]

    def compoute_boresights(self):
        img1 = self.get_image_object(0)
        img2 = self.get_image_object(1)

        igm_1 = ortho_tools.create_igm_image(img1, self._dem)
        igm_2 = ortho_tools.create_igm_image(img2, self._dem)

        img1_ortho = ortho_tools.create_full_ortho_gtiff_image(igm_1, dem=self._dem)
        img2_ortho = ortho_tools.create_full_ortho_gtiff_image(igm_2, dem=self._dem)

        gtiff1 = img1_ortho.get_image_band(0)
        gtiff2 = img2_ortho.get_image_band(0)

        gtiff1 = image_utils.normalize_grayscale_image(gtiff1, 0, 255, numpy.uint8)
        gtiff2 = image_utils.normalize_grayscale_image(gtiff2, 0, 255, numpy.uint8)

        # Initiate SIFT detector
        sift = cv.xfeatures2d_SIFT.create()
        # find the keypoints and descriptors with SIFT

        kp1, des1 = sift.detectAndCompute(gtiff1, None)
        kp2, des2 = sift.detectAndCompute(gtiff2, None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i] = [1, 0]
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=cv.DrawMatchesFlags_DEFAULT)
        img3 = cv.drawMatchesKnn(gtiff1, kp1, gtiff2, kp2, matches, None, **draw_params)
        plt.imshow(img3)
        plt.show()
