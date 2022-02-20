import os
import numpy
from resippy.atmospheric_compensation.arm_climate_model import ArmClimateModel
from resippy.utils import video_utils
from resippy.utils import file_utils
import imageio

output_dir = os.path.expanduser("~/Downloads/arm_site_translation")


def animate_arm_site_translation_uv_image():
    arm_png_path = os.path.expanduser("~/Data/SMART/arm_data/sgpirsiviscldmaskC1.a1.20190714.172247.jpg.2019-07-14T17-22-47_RBratio_N2_mask.png")
    file_utils.make_dir_if_not_exists(output_dir)

    cloud_deck_height = 1000
    max_translation = 1000
    n_steps = 50
    translation_steps = numpy.linspace(0, max_translation, n_steps)

    arm_model = ArmClimateModel.from_image_file(arm_png_path)
    arm_model.center_xyz_location = (0, 0, 0)
    for i, translation in enumerate(translation_steps):
        print("step " + str(i) + " of " + str(len(translation_steps)))
        t = int(translation)
        new_model = ArmClimateModel.translated_arm_model(arm_model, cloud_deck_height, (translation, 0, 0))
        t_string = str(t).zfill(10)
        fname = "arm_site_translated_" + t_string + ".png"
        output_fullpath = os.path.join(output_dir, fname)
        imageio.imwrite(output_fullpath, new_model.uv_image)


def main():
    animate_arm_site_translation_uv_image()
    output_fname = os.path.expanduser("~/Downloads/arm_animation.mp4")
    video_utils.directory_of_images_to_mp4(output_dir, 10, output_fname)


if __name__ == '__main__':
    main()
