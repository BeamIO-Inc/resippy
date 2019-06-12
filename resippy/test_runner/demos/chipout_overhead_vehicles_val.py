import os
from resippy.test_runner import demo_data_base_dir
from resippy.test_runner import demo_data_save_dir
from resippy.utils import file_utils as file_utils
from resippy.utils.image_utils import image_mask_utils
from resippy.utils.image_utils import image_chipper
from resippy.utils import numpy_and_array_utils
from imageio import imread


png_fullpath = file_utils.get_path_from_subdirs(demo_data_base_dir,
                                                ["image_data",
                                                 "overhead_vehicle_data",
                                                 "Potsdam_ISPRS",
                                                 "top_potsdam_6_8_RGB.png"])

png_positive_mask_fullpath = file_utils.get_path_from_subdirs(demo_data_base_dir,
                                                ["image_data",
                                                 "overhead_vehicle_data",
                                                 "Potsdam_ISPRS",
                                                 "top_potsdam_6_8_RGB_Annotated_Cars.png"
                                                 ])

png_negative_mask_fullpath = file_utils.get_path_from_subdirs(demo_data_base_dir,
                                                ["image_data",
                                                 "overhead_vehicle_data",
                                                 "Potsdam_ISPRS",
                                                 "top_potsdam_6_8_RGB_Annotated_Negatives.png"
                                                 ])

save_dir_base = file_utils.get_path_from_subdirs(demo_data_save_dir, ["overhead_vehicles", "testing_data"])
file_utils.make_dir_if_not_exists(save_dir_base)


def create_vehicle_training_chips():
    print("CHIPPING OUT VEHICLES FROM EVALUATION IMAGE, THIS MIGHT TAKE A WHILE")
    print("save dir = " + save_dir_base)

    chipper_ny = 50
    chipper_nx = 50

    output_chip_ny = 244
    output_chip_nx = 244
    vehicle_mask = imread(png_positive_mask_fullpath)
    not_vehicle_mask = imread(png_negative_mask_fullpath)
    image_data = imread(png_fullpath)
    vehicle_locs_y, vehicle_locs_x = image_mask_utils.rgb_image_point_mask_to_pixel_locations(vehicle_mask)

    # remove chips that aren't very vehicle like, or have vehicles that are badly obscured.
    vehicle_yx_to_remove = [(1583, 1650),
                            (1599, 1217),
                            (2094, 122),
                            (2106, 160)]
    vehicle_loc_tuples = numpy_and_array_utils.separate_lists_to_list_of_tuples([vehicle_locs_y, vehicle_locs_x])
    for yx_to_remove in vehicle_yx_to_remove:
        vehicle_loc_tuples.remove(yx_to_remove)
    vehicle_locs_y, vehicle_locs_x = numpy_and_array_utils.list_of_tuples_to_separate_lists(vehicle_loc_tuples)

    # chip out the vehicles
    vehicle_chips, vehicle_upper_lefts = image_chipper.\
        chip_images_by_pixel_centers(image_data,
                                     vehicle_locs_y,
                                     vehicle_locs_x,
                                     chip_ny_pixels=chipper_ny,
                                     chip_nx_pixels=chipper_nx)

    not_vehicle_locs_y, not_vehicle_locs_x = image_mask_utils.rgb_image_point_mask_to_pixel_locations(not_vehicle_mask)

    # remove chips that do actually contain vehicles.
    not_vehicle_yx_to_remove = [(489, 1124),
                                (584, 1186),
                                (1222, 84),
                                (1271, 1409),
                                (1278, 1421),
                                (1487, 1497),
                                (1499, 1883),
                                (1531, 1444),
                                (1597, 1820),
                                (1601, 1766),
                                (1659, 922),
                                (1677, 811),
                                (1694, 780)]
    not_vehicle_loc_tuples = numpy_and_array_utils.separate_lists_to_list_of_tuples([not_vehicle_locs_y, not_vehicle_locs_x])
    for yx_to_remove in not_vehicle_yx_to_remove:
        not_vehicle_loc_tuples.remove(yx_to_remove)
    not_vehicle_locs_y, not_vehicle_locs_x = numpy_and_array_utils.list_of_tuples_to_separate_lists(not_vehicle_loc_tuples)

    not_vehicle_chips, not_vehicle_upper_lefts = image_chipper.\
        chip_images_by_pixel_centers(image_data,
                                     not_vehicle_locs_y,
                                     not_vehicle_locs_x,
                                     chip_ny_pixels=chipper_ny,
                                     chip_nx_pixels=chipper_nx)

    vehicles_save_dir = os.path.join(save_dir_base, "vehicle")
    not_vehicles_save_dir = os.path.join(save_dir_base, "not_vehicle")
    file_utils.make_dir_if_not_exists(vehicles_save_dir)
    file_utils.make_dir_if_not_exists(not_vehicles_save_dir)

    # create filenames for vehicles and non-vehicles based on original center locations.  This will let us
    # look through our chips for false positives and remove them from the masks
    vehicle_chips_fnames_list = ["vehicle_center_y_" + str(vehicle_locs_y[i]) + "_center_x_" + str(vehicle_locs_x[i]) for i in range(len(vehicle_locs_x))]
    not_vehicle_chips_fnames_list = ["not_vehicle_center_y_" + str(not_vehicle_locs_y[i]) + "_center_x_" + str(not_vehicle_locs_x[i]) for i in range(len(not_vehicle_locs_x))]

    image_chipper.write_chips_to_disk(vehicle_chips, vehicles_save_dir,
                                      fnames_list=vehicle_chips_fnames_list,
                                      output_chip_ny=output_chip_ny, output_chip_nx=output_chip_nx)

    image_chipper.write_chips_to_disk(not_vehicle_chips, not_vehicles_save_dir,
                                      fnames_list=not_vehicle_chips_fnames_list,
                                      output_chip_ny=output_chip_ny, output_chip_nx=output_chip_nx)

    print("DONE CHIPPING OUT IMAGES")

def main():
    create_vehicle_training_chips()


if __name__ == "__main__":
    main()
