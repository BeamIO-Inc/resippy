import resippy.utils.image_utils.image_utils as image_utils
import resippy.utils.file_utils as file_utils
from resippy.test_runner import demo_data_base_dir
from resippy.test_runner import demo_data_save_dir
import imageio
import os
import numpy as np

save_dir = os.path.join(demo_data_save_dir, "colormap_demos")
file_utils.make_dir_if_not_exists(save_dir)

png_fullpath = file_utils.get_path_from_subdirs(demo_data_base_dir,
                                                ["image_data",
                                                 "overhead_vehicle_data",
                                                 "Potsdam_ISPRS",
                                                 "top_potsdam_2_14_RGB.png"])


def main():
    image_data = imageio.imread(png_fullpath)
    image_data = image_data[:, :, 0:3]
    grayscale_image = image_data[:, :, 0]
    colormapped_image = image_utils.apply_colormap_to_grayscale_image(grayscale_image)
    imageio.imwrite(os.path.join(save_dir, "colormapped_image.png"), colormapped_image)
    blended_image = np.asarray(image_utils.blend_images(image_data, colormapped_image), dtype=np.uint8)
    imageio.imwrite(os.path.join(save_dir, "blended_image.png"), blended_image)


if __name__ == '__main__':
    main()
