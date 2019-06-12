import os
from resippy.image_objects.image_factory import ImageFactory
from resippy.test_runner import demo_data_base_dir
from resippy.test_runner import demo_data_save_dir
from resippy.utils import file_utils as file_utils


micasense_dir = file_utils.get_path_from_subdirs(demo_data_base_dir, ['image_data',
                                                                      'multispectral',
                                                                      'micasense',
                                                                      '20181019_hana',
                                                                      '1703',
                                                                      'micasense',
                                                                      'processed'
                                                                      ])

gtiff_full_path = file_utils.get_path_from_subdirs(micasense_dir,
                                                ['pix4d_1703_mica_imgs85_1607',
                                                 '3_dsm_ortho',
                                                 'pix4d_1704_mica_85_1607_dsm.tif'])

save_dir = os.path.join(demo_data_save_dir, "gtiff_demos")
file_utils.make_dir_if_not_exists(save_dir)


def create_maptiles():
    print("")
    print("CREATE MAPTILES DEMO")
    file_utils.make_dir_if_not_exists(save_dir)
    gtiff_image_object = ImageFactory.geotiff.from_file(gtiff_full_path)
    gtiff_image_object.create_map_tiles("/tmp/")


def gtiff_dsm_image_object_save():
    print("")
    print("GTIFF DSM IMAGE OBJECT SAVE DEMO")
    file_utils.make_dir_if_not_exists(save_dir)
    gtiff_image_object = ImageFactory.geotiff.from_file(gtiff_full_path)
    save_fname = "gtiff_dsm_write_to_disk.tif"
    save_fullpath = os.path.join(save_dir, save_fname)
    gtiff_image_object.write_to_disk(save_fullpath)
    print("geotiff written to: " + save_fullpath)


def gtiff_dsm_read_then_create_from_numpy_array():
    """
    This demonstrates how to read a geotiff file in, add a bias term, update the nodata values, and write out to disk
    """
    print("")
    print("CREATE GEOTIFF FROM NUMPY ARRAY DEMO")
    file_utils.make_dir_if_not_exists(save_dir)
    gtiff_image_object = ImageFactory.geotiff.from_file(gtiff_full_path)
    save_fname = "gtiff_dsm_ortho_from_numpy.tif"
    numpy_save_fullpath = os.path.join(save_dir, save_fname)
    gtiff_from_numpy = ImageFactory.geotiff.from_numpy_array(gtiff_image_object.read_all_image_data_from_disk(),
                                                             gtiff_image_object.get_point_calculator().get_geot(),
                                                             gtiff_image_object.get_point_calculator().get_projection(),
                                                             gtiff_image_object.get_metadata().get_nodata_val())
    gtiff_from_numpy.write_to_disk(numpy_save_fullpath)
    numpy_saved_image = ImageFactory.geotiff.from_file(numpy_save_fullpath)
    data = numpy_saved_image.read_all_image_data_from_disk()
    val_to_add = 10
    data = data + val_to_add
    save_fname = "gtiff_dsm_ortho_from_numpy_plus_ten.tif"
    numpy_save_fullpath = os.path.join(save_dir, save_fname)
    gtiff_from_numpy.set_image_data(data)
    gtiff_from_numpy.get_metadata().set_nodata_val(gtiff_image_object.get_metadata().get_nodata_val() + val_to_add)
    gtiff_from_numpy.write_to_disk(numpy_save_fullpath)
    print("geotiff written to: " + numpy_save_fullpath)


def main():
    # TODO leave this out until more documentation can be provided on how to reliably install gdal2tiles
    #create_maptiles()
    gtiff_dsm_image_object_save()
    gtiff_dsm_read_then_create_from_numpy_array()


if __name__ == '__main__':
    main()
