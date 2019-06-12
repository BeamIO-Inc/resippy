from resippy.test_runner.demos import gtiff_demos
from resippy.test_runner.demos import micasense_ortho_demo
from resippy.test_runner.demos import colormap_image_demo
from resippy.test_runner.demos import digital_globe_demos
from resippy.test_runner.demos import envi_image_demos
from resippy.test_runner.demos import mutual_information_demo
from resippy.test_runner.demos import pixels_obstructed_by_dem_demo
from resippy.test_runner.demos import chipout_overhead_vehicles_train


def main():
    gtiff_demos.main()
    micasense_ortho_demo.main()
    colormap_image_demo.main()
    digital_globe_demos.main()
    envi_image_demos.main()
    mutual_information_demo.main()
    pixels_obstructed_by_dem_demo.main()
    chipout_overhead_vehicles_train.main()


if __name__ == '__main__':
    main()

