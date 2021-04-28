import os
from resippy.utils import envi_utils


header_dir = os.path.expanduser("~/Data/resippy_demo_data/envi_headers")
header_fullpath = os.path.join(header_dir, "envi_example_header_1.hdr")
output_fullpath = os.path.expanduser("~/Downloads/envi_header_tmp.hdr")

input_header = envi_utils.read_envi_header(header_fullpath)
input_header['lines'] = 5500

envi_utils.write_envi_header(input_header, output_fullpath)

header2 = envi_utils.read_envi_header(output_fullpath)
stop = 1