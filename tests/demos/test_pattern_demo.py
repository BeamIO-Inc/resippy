from resippy.utils.image_utils import test_pattern_generator

gradient_image = test_pattern_generator.gradient_image(100, 100,  ul_val=1, mid_val=0, br_val=1)
red_to_blue_image = test_pattern_generator.red_to_blue(100, 100)
tmp = test_pattern_generator.test_pattern(100, 100)