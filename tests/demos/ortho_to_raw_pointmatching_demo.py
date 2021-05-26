from resippy.utils.envi_utils import read_envi_header
from resippy.utils.envi_utils import read_envi_image
from resippy.image_objects.envi.envi_image_factory import EnviImageFactory
from resippy.image_objects.earth_overhead.igm.igm_image import IgmImage
import os

raw_dir = os.path.expanduser("~/Data/hyperspec3/raw")
ortho_dir = os.path.expanduser("~/Data/hyperspec3/ortho")

envi_ortho_header_fname = os.path.join(ortho_dir, "raw_1440_0-0-0-0_or.hdr")
envi_ortho_fname = os.path.join(ortho_dir, "raw_1440_0-0-0-0_or")

envi_raw_header_fname = os.path.join(raw_dir, "raw_1440.hdr")
envi_raw_fname = os.path.join(raw_dir, "raw_1400")

envi_ortho_header = read_envi_header(envi_ortho_header_fname)
mapinfo_fields = envi_ortho_header['map info'].split(',')
lon_ul = float(mapinfo_fields[3])
lat_ul = float(mapinfo_fields[4])
lon_gsd = float(mapinfo_fields[5])
lat_gsd = float(mapinfo_fields[6])

geot = (lon_ul, lon_gsd, 0, lat_ul, 0, -lat_gsd)

envi_ortho_image = read_envi_image(envi_ortho_fname, envi_ortho_header_fname)

ul_ortho_kp_match = (0, 0)
br_ortho_kp_match = (envi_ortho_header['lines']-1, envi_ortho_header['samples']-1)

raw_image_object = EnviImageFactory.from_file(envi_raw_fname, envi_raw_header_fname)

# igm_image = IgmImage.from_params(envi_ortho_image,)

stop = 1