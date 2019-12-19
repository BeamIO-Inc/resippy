from resippy.test_runner import demo_data_base_dir, demo_data_save_dir

from resippy.utils import file_utils as file_utils
from resippy.utils import envi_utils
from resippy.image_objects.image_factory import ImageFactory
import os


radiance_cube_fname = file_utils.get_path_from_subdirs(demo_data_base_dir, ["image_data",
                                                                            "hyperspectral",
                                                                            "cooke_city",
                                                                            "self_test",
                                                                            "HyMap",
                                                                            "self_test_rad.img"])

save_dir = os.path.join(demo_data_save_dir, "envi_image_demos")
file_utils.make_dir_if_not_exists(save_dir)
save_image_fname = os.path.join(save_dir, "cooke_city_cube.img")


def open_and_save_envi_image():
    print("")
    print("OPEN AND SAVE ENVI IMAGE DEMO")
    envi_image = ImageFactory.envi.from_file(radiance_cube_fname)
    image_data = envi_image.read_all_image_data_from_disk()
    envi_utils.write_envi_image(image_data, save_image_fname)
    print("saved envi cube to: " + save_image_fname)


def main():
    open_and_save_envi_image()


if __name__ == '__main__':
    main()
