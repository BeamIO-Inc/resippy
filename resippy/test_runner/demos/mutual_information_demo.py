import imageio
from resippy.test_runner import demo_data_base_dir, demo_data_save_dir
import resippy.utils.image_utils.image_registration as imreg
from resippy.utils import file_utils
import resippy.utils.image_utils.image_chipper as image_chipper
from resippy.utils.image_utils import image_utils

import numpy as np
from numpy import ndarray
import os
import logging

save_dir = os.path.join(demo_data_save_dir, "mmi_demos")
file_utils.make_dir_if_not_exists(save_dir)

micasense_dir = file_utils.get_path_from_subdirs(demo_data_base_dir, ['image_data',
                                                                      'multispectral',
                                                                      'micasense',
                                                                      '20181019_hana',
                                                                      '1703',
                                                                      'micasense',
                                                                      'processed'
                                                                      ])
band_fname_dict = {}
for cam_num in range(1, 6):
    band_num = str(cam_num)
    band_fname_dict['band' + band_num] = \
        file_utils.get_path_from_subdirs(micasense_dir, ['merged', 'L0_IMG_0404_' + band_num + '.tif'])


def mmi_window_demo():
    print("")
    print("STARTING A SLIDING WINDOW MUTUAL INFORMATION DEMO, THIS MIGHT TAKE A LITTLE BIT...")
    max_pixel_shift = 50
    n_steps = 10

    im1 = imageio.imread(band_fname_dict['band1'])      # type: ndarray
    im2 = imageio.imread(band_fname_dict['band2'])      # type: ndarray

    ny_original, nx_original = im1.shape
    chip_size_y = ny_original - max_pixel_shift*2 - 10
    chip_size_x = nx_original - max_pixel_shift*2 - 10

    image_center_y = int(ny_original / 2)
    image_center_x = int(nx_original / 2)
    im1_chip, image1_ul = image_chipper.chip_images_by_pixel_centers(im1,
                                                                     [image_center_y],
                                                                     [image_center_x],
                                                                     chip_size_y,
                                                                     chip_size_x)

    im2_chip_original, image2_ul = image_chipper.chip_images_by_pixel_centers(im2,
                                                                              [image_center_y],
                                                                              [image_center_x],
                                                                              chip_size_y,
                                                                              chip_size_x)

    im1_chip = im1_chip[0, :, :]
    im2_chip_original = im2_chip_original[0, :, :]
    # im1_chip = ndimage.gaussian_filter(im1_chip, sigma=10)

    mmi_scores = []
    # generate pixel centers for chipping image 2
    x_centers = np.linspace(image_center_x - max_pixel_shift, image_center_x + max_pixel_shift - 1, n_steps)
    y_centers = np.linspace(image_center_y - max_pixel_shift, image_center_y + max_pixel_shift - 1, n_steps)

    pixel_x_centers_list = []
    pixel_y_centers_list = []
    for y in y_centers:
        for x in x_centers:
            pixel_y_centers_list.append(y)
            pixel_x_centers_list.append(x)

    im2_chips, image2_uls = image_chipper.chip_images_by_pixel_centers(im2,
                                                                          pixel_y_centers_list,
                                                                          pixel_x_centers_list,
                                                                          chip_size_y,
                                                                          chip_size_x)
    for i, im2_chip in enumerate(im2_chips):
        mmi_score = imreg.MMI.image_mmi(im1_chip, im2_chip)
        mmi_scores.append(mmi_score)
        logging.debug("frame: " + str(i) + " of " + str(len(im2_chips)))
    mmi_scores_image = np.reshape(mmi_scores, (n_steps, n_steps))
    image_fname = os.path.join(save_dir, "mmi_scores_image.png")
    colormapped_mmi_scores = image_utils.apply_colormap_to_grayscale_image(mmi_scores_image)
    imageio.imwrite(image_fname, colormapped_mmi_scores)

    imageio.imwrite(os.path.join(save_dir, "im1.png"), im1_chip)
    imageio.imwrite(os.path.join(save_dir, "im2_unreg.png"), im2_chip_original)

    highest_mmi_score_index = np.where(mmi_scores == np.max(mmi_scores))[0]
    im2_chip_reg = im2_chips[highest_mmi_score_index][0]
    imageio.imwrite(os.path.join(save_dir, "im2_reg.png"), im2_chip_reg)
    print("MMI SLIDING WINDOW TEST PASSED. RESULTS IN " + str(save_dir))


def main():
    mmi_window_demo()


if __name__ == '__main__':
    main()
