from __future__ import division

import unittest
from resippy.utils.image_utils import image_utils as image_utils
import resippy.photogrammetry.crs_defs as crs_defs
from resippy.image_objects.image_factory import ImageFactory
import numpy as np
import osgeo.gdal_array as gdal_array
import osr


class TestGeotiffImages(unittest.TestCase):

    def test_uniform_image(self):
        npix_x = 1000
        npix_y = 800
        nbands = 3
        geot = [0, 1, 0, 0, 0, -1]
        image_data = image_utils.create_uniform_image_data(npix_x, npix_y, nbands)
        gtiff_image = ImageFactory.geotiff.from_numpy_array(image_data, geot, crs_defs.PROJ_4326)
        assert (gtiff_image.get_image_data() == image_data).all()
        assert gtiff_image.get_point_calculator().get_geot() == geot
        assert gtiff_image.get_metadata().get_npix_x() == npix_x
        assert gtiff_image.get_metadata().get_npix_y() == npix_y
        print("image from from numpy array test passed")

    def test_datatypes(self):
        datatypes = [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.float32, np.float64]
        npix_x = 100
        npix_y = 80
        nbands = 1
        output_fname = "/tmp/test_geotiff.tif"
        geot = [0, 1, 0, 0, 0, -1]
        for datatype in datatypes:
            image_data = image_utils.create_uniform_image_data(npix_x, npix_y, nbands, dtype=datatype)
            gtiff_image = ImageFactory.geotiff.from_numpy_array(image_data, geot, crs_defs.PROJ_4326)
            gtiff_image.write_to_disk(output_fname)
            gtiff_image_from_file = ImageFactory.geotiff.from_file(output_fname)
            gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(datatype)

            assert gtiff_image.get_metadata().get_gdal_datatype() == gtiff_image_from_file.get_metadata().get_gdal_datatype() == gdal_dtype
        print("datatype tests passed")

    def test_constructors(self):
        npix_x = 1000
        npix_y = 800
        nbands = 3
        output_fname = "/tmp/test_geotiff.tif"
        geot = [0, 1, 0, 0, 0, -1]
        image_data = image_utils.create_uniform_image_data(npix_x, npix_y, nbands)
        gtiff_image = ImageFactory.geotiff.from_numpy_array(image_data, geot, crs_defs.PROJ_4326)
        gtiff_data = gtiff_image.get_image_data()
        gtiff_image.write_to_disk(output_fname)
        gtiff_image_from_file = ImageFactory.geotiff.from_file(output_fname)
        assert gtiff_image.get_metadata().metadata_dict == gtiff_image_from_file.get_metadata().metadata_dict
        assert (gtiff_data == gtiff_image_from_file.read_all_image_data_from_disk()).all()

        numpy_osr = osr.SpatialReference()
        numpy_osr.ImportFromProj4(gtiff_image.get_point_calculator().get_projection().srs)
        from_disk_osr = osr.SpatialReference()
        from_disk_osr.ImportFromProj4(gtiff_image_from_file.get_point_calculator().get_projection().srs)
        numpy_osr_wkt = numpy_osr.ExportToWkt()
        from_disk_osr_wkt = from_disk_osr.ExportToWkt()
        assert numpy_osr_wkt == from_disk_osr_wkt


if __name__ == '__main__':
    unittest.main()
