import os
from resippy.atmospheric_compensation.arm_climate_model import ArmClimateModel

import matplotlib.pyplot as plt


def load_arm_data():
    arm_png_path = os.path.expanduser("~/Data/SMART/arm_data/sgpirsiviscldmaskC1.a1.20190714.172247.jpg.2019-07-14T17-22-47_RBratio_N2_mask.png")
    arm_model = ArmClimateModel.from_image_file(arm_png_path)
    arm_model.center_xyz_location = (0, 0, 0)
    new_model = ArmClimateModel.translated_arm_model(arm_model, 1000, (1000, 0, 0))
    stop = 1


def main():
    load_arm_data()


if __name__ == '__main__':
    main()
