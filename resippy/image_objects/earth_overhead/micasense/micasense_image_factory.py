from __future__ import division

import pyproj

from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.pix4d_point_calc import Pix4dPointCalc
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.physical_model_point_calc \
    import PhysicalModelPointCalc
from resippy.image_objects.earth_overhead.micasense.micasense_image import MicasenseImage
from resippy.image_objects.earth_overhead.micasense.micasense_metadata import MicasenseMetadata
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.earth_overhead_sensor_model \
    import EarthOverheadSensorModel

import os
from uuid import uuid4
from PIL import Image
import numpy as np
import json


class MicasenseImageFactory:
    @staticmethod
    def _from_point_calcs(band_fname_dict,
                          point_calcs,
                          ):
        sensor_model = EarthOverheadSensorModel()
        sensor_model.set_point_calcs(point_calcs)
        sensor_model.set_approximate_lon_lat_center(*point_calcs[0].get_approximate_lon_lat_center())
        sensor_model.set_projection(point_calcs[0].get_projection())
        sensor_model._bands_coregistered = False

        metadata = MicasenseMetadata()
        basename, _ = os.path.splitext(band_fname_dict['band1'])
        fparts = basename.split('_')
        fparts.pop()
        name = "_".join(fparts)
        metadata.set_image_name(name)
        metadata.set_nodata_val(0)

        micasense_image = MicasenseImage()
        micasense_image.band_fnames.append(band_fname_dict['band1'])
        micasense_image.band_fnames.append(band_fname_dict['band2'])
        micasense_image.band_fnames.append(band_fname_dict['band3'])
        micasense_image.band_fnames.append(band_fname_dict['band4'])
        micasense_image.band_fnames.append(band_fname_dict['band5'])

        micasense_image.set_metadata(metadata)
        micasense_image.set_point_calculator(sensor_model)

        return micasense_image

    @staticmethod
    def from_image_number_and_pix4d(band_fname_dict,            # type: dict
                                    pix4d_master_params_dict    # type: dict
                                    ):                          # type: (...) -> MicasenseImage
        def create_point_calc(band_fname, params):
            point_calc = Pix4dPointCalc.init_from_params(band_fname, params)
            point_calc.reverse_x_pixels = True
            point_calc.reverse_y_pixels = True
            return point_calc

        point_calc_1 = create_point_calc(band_fname_dict['band1'], pix4d_master_params_dict)
        point_calc_2 = create_point_calc(band_fname_dict['band2'], pix4d_master_params_dict)
        point_calc_3 = create_point_calc(band_fname_dict['band3'], pix4d_master_params_dict)
        point_calc_4 = create_point_calc(band_fname_dict['band4'], pix4d_master_params_dict)
        point_calc_5 = create_point_calc(band_fname_dict['band5'], pix4d_master_params_dict)

        return MicasenseImageFactory._from_point_calcs(band_fname_dict, [point_calc_1, point_calc_2, point_calc_3,
                                                                         point_calc_4, point_calc_5])

    @staticmethod
    def from_image_number_and_json_fname(band_fname_dict,   # type: dict
                                         json_fname         # type: str
                                         ):                 # type: (...) -> MicasenseImage
        x = [334265.746789, 334265.717233, 334265.720081, 334265.749637, 334265.733435]
        y = [4748702.207249, 4748702.203111, 4748702.182314, 4748702.186452, 4748702.194782]

        def create_point_calc(band_fname, params):
            band_num = int(band_fname[band_fname.rfind('.')-1])

            intrinsic_list = params['intrinsic']
            intrinsic_params = [ip for ip in intrinsic_list if ip['band_number'] == band_num][0]

            point_calc = PhysicalModelPointCalc.init_from_params_and_center(intrinsic_params, params['extrinsic'],
                                                                            x[band_num-1], y[band_num-1])

            point_calc.reverse_x_pixels = True
            point_calc.reverse_y_pixels = True
            return point_calc

        with open(json_fname) as json_file:
            json_data = json.load(json_file)

        params = json_data['parameters']

        point_calc_1 = create_point_calc(band_fname_dict['band1'], params)
        point_calc_2 = create_point_calc(band_fname_dict['band2'], params)
        point_calc_3 = create_point_calc(band_fname_dict['band3'], params)
        point_calc_4 = create_point_calc(band_fname_dict['band4'], params)
        point_calc_5 = create_point_calc(band_fname_dict['band5'], params)

        x, y = point_calc_1.lon_lat_alt_to_pixel_x_y(0, 0, 100)
        print("x: " + str(x))
        print("y: " + str(y))

        return MicasenseImageFactory._from_point_calcs(band_fname_dict, [point_calc_1, point_calc_2, point_calc_3,
                                                                         point_calc_4, point_calc_5])

    @staticmethod
    def from_numpy_and_pix4d(image_data,            # type: np.ndarray
                             metadata,              # type: MicasenseMetadata
                             point_calculator,      # type: EarthOverheadSensorModel
                             working_dir=None,      # type: str
                             band_basenames=None,   # type: list
                             ):                     # type: (...) -> MicasenseImage
        def write_tmp_band(band_data, working_dir=None, band_basename=None):
            uname = os.path.basename(os.getenv("HOME"))

            if working_dir is None:
                output_dir = os.path.join("/tmp", uname)
                os.makedirs(output_dir, exist_ok=True)
            else:
                output_dir = working_dir

            if band_basename is None:
                band_basename = str(uuid4().hex) + ".tif"
            output_fname = os.path.join(output_dir, band_basename)
            out_img = Image.fromarray(band_data)
            out_img.save(output_fname)

            return output_fname

        micasense_image = MicasenseImage()

        ny_nx_nbands = np.shape(image_data)
        nbands = ny_nx_nbands[2]
        for band in range(nbands):
            if band_basenames is not None:
                band_basename = band_basenames[band]
            else:
                band_basename = None
            fn = write_tmp_band(np.squeeze(image_data[:,:,band]), working_dir, band_basename)
            micasense_image.band_fnames.append(fn)

        micasense_image.set_metadata(metadata)
        micasense_image.set_point_calculator(point_calculator)

        return micasense_image
