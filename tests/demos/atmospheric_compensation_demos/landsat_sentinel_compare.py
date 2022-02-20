import os
from resippy.image_objects.image_factory import ImageFactory


aker_fullpath = "/Users/jasoncasey/Data/aker/merged.tif"
aker_gtiff = ImageFactory.geotiff.from_file(aker_fullpath)

ny = aker_gtiff.metadata.metadata_dict['npix_y']
nx = aker_gtiff.metadata.metadata_dict['npix_x']

scaled_reflectance_data = aker_gtiff.read_all_image_data_from_disk()

landsat_multiply = 0.000027
landsat_add = -0.2

sentintel_multiply = 0.0001
sentinel_add = 0

# HERE is where you would replace the "multiply" and "add" variables depending on lansat vs sentinel
reflectance_data = scaled_reflectance_data*sentintel_multiply + sentinel_add

geot = aker_gtiff.pointcalc._geo_t
projection = aker_gtiff.pointcalc.get_projection()
new_geotiff = ImageFactory.geotiff.from_numpy_array(reflectance_data, geot, projection)

output_path = os.path.expanduser("~/Downloads/aker_reflectance.tif")

new_geotiff.write_to_disk(output_path)
