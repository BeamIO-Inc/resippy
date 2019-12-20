import resippy.photogrammetry.ortho_tools as ortho_tools
from resippy.image_objects.image_factory import ImageFactory
from resippy.test_runner import demo_data_base_dir, demo_data_save_dir
from resippy.utils import file_utils as file_utils

import os

import logging

dg_dir = file_utils.get_path_from_subdirs(demo_data_base_dir,
                                          ["image_data",
                                           "digital_globe",
                                           "Tripoli_View-Ready_Stereo_8_Band_Bundle_30cm",
                                           "056082094030"])

multispectral_image_fname_1 = file_utils.get_path_from_subdirs(dg_dir,
                                                               ["056082094030_01_P001_MUL",
                                                                "15AUG31100732-M2AS-056082094030_01_P001.TIF"])

save_dir = os.path.join(demo_data_save_dir, "digital_globe_demos")
file_utils.make_dir_if_not_exists(save_dir)

# make the image object object
digital_globe_image = ImageFactory.digital_globe.view_ready_stereo.from_file(multispectral_image_fname_1)


def ortho_digital_globe_image():
    print("STARTING A DIGITAL GLOBE DEMO")
    nx = 1000
    ny = 1000

    extent = ortho_tools.get_extent(digital_globe_image)

    orthod_image = ortho_tools.create_ortho_gtiff_image_world_to_sensor(digital_globe_image, nx, ny, extent)

    save_fname = os.path.join(save_dir, "dg_image_ortho_15AUG31100732-M2AS-056082094030_01_P001.tif")
    orthod_image.write_to_disk(save_fname)

    logging.debug("created ortho from digital globe view ready stereo image.")
    print("ortho image located at: " + save_fname)


def main():
    ortho_digital_globe_image()


if __name__ == '__main__':
    main()
