from resippy.atmospheric_compensation.arm_climate_model import ArmClimateModel


def load_arm_data():
    arm_png_path = "/Users/jkc/Data/SMART/arm_data/sgpirsiviscldmaskC1.a1.20190714.172247.jpg.2019-07-14T17-22-47_RBratio_N2_mask.png"
    arm_model = ArmClimateModel.from_image_file(arm_png_path)


def main():
    load_arm_data()


if __name__ == '__main__':
    main()
