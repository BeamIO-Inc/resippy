from __future__ import division

from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.pix4d_point_calc import Pix4dPointCalc
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.opencv_point_calc \
    import OpenCVPointCalc
from resippy.image_objects.earth_overhead.micasense.micasense_image import MicasenseImage
from resippy.image_objects.earth_overhead.micasense.micasense_metadata import MicasenseMetadata
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.earth_overhead_sensor_model \
    import EarthOverheadSensorModel
from resippy.utils.micasense.micasense_metadata import Metadata
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.ideal_pinhole_fpa_local_utm_point_calc import IdealPinholeFpaLocalUtmPointCalc

import os
from uuid import uuid4
from PIL import Image
import numpy as np


class MicasenseImageFactory:
    @staticmethod
    def _from_point_calcs(band_fname_dict,
                          point_calcs,
                          ):
        sensor_model = EarthOverheadSensorModel()
        sensor_model.set_point_calcs(point_calcs)
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

        micasense_image = MicasenseImageFactory._from_point_calcs(band_fname_dict, [point_calc_1, point_calc_2,
                                                                                    point_calc_3, point_calc_4,
                                                                                    point_calc_5])

        sensor_model = micasense_image.get_point_calculator()
        sensor_model.set_approximate_lon_lat_center(*sensor_model.get_point_calcs()[0].get_approximate_lon_lat_center())

        return micasense_image

    @staticmethod
    def from_image_number_and_opencv(band_fname_dict,   # type: dict
                                     opencv_params      # type: dict
                                     ):                 # type: (...) -> MicasenseImage
        def create_point_calc(band_fname, params):
            band_num = int(band_fname[band_fname.rfind('.')-1])

            intrinsic_list = params['intrinsic']
            intrinsic_params = [ip for ip in intrinsic_list if ip['band_number'] == band_num][0]

            point_calc = OpenCVPointCalc.init_from_params(intrinsic_params, params['extrinsic'])

            point_calc.reverse_x_pixels = True
            point_calc.reverse_y_pixels = True
            return point_calc

        point_calc_1 = create_point_calc(band_fname_dict['band1'], opencv_params)
        point_calc_2 = create_point_calc(band_fname_dict['band2'], opencv_params)
        point_calc_3 = create_point_calc(band_fname_dict['band3'], opencv_params)
        point_calc_4 = create_point_calc(band_fname_dict['band4'], opencv_params)
        point_calc_5 = create_point_calc(band_fname_dict['band5'], opencv_params)

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

    @staticmethod
    def quick_preview_frome_fname(base_fname,  # type: dict
                                  ):  # type: (...) -> MicasenseImage
        stop = 1
        camera_fnames = [base_fname + "_1.tif",
                         base_fname + "_2.tif",
                         base_fname + "_3.tif",
                         base_fname + "_4.tif",
                         base_fname + "_5.tif"]
        point_calcs = []
        for camera_fname in camera_fnames:
            camera_metadata = Metadata(camera_fname)
            lon = camera_metadata.exif['Composite:GPSLongitude']
            lat = camera_metadata.exif['Composite:GPSLatitude']
            alt = camera_metadata.exif['Composite:GPSAltitude']
            omega = float(camera_metadata.exif['XMP:Roll'])
            phi = float(camera_metadata.exif['XMP:Pitch'])
            kappa = float(camera_metadata.exif['XMP:Yaw'])
            npix_x = camera_metadata.exif['EXIF:ImageWidth']
            npix_y = camera_metadata.exif['EXIF:ImageHeight']
            pixel_pitch_x = 3.75
            pixel_pitch_y = 3.75
            focal_length = camera_metadata.exif['XMP:PerspectiveFocalLength']
            point_calc = IdealPinholeFpaLocalUtmPointCalc.init_from_wgs84_params(lon, lat, alt, omega, phi, -kappa, npix_x, npix_y, pixel_pitch_x, pixel_pitch_y, focal_length, alt_units='feet', flip_y=True)
            point_calcs.append(point_calc)

        sensor_model = EarthOverheadSensorModel()
        sensor_model.set_point_calcs(point_calcs)
        sensor_model.set_projection(point_calcs[0].get_projection())
        sensor_model._bands_coregistered = False
        native_lon, native_lat = point_calcs[0].get_approximate_lon_lat_center()
        sensor_model.set_approximate_lon_lat_center(native_lon, native_lat)

        micasense_image = MicasenseImage()
        micasense_image.band_fnames = camera_fnames

        metadata = MicasenseMetadata()
        micasense_image.set_metadata(metadata)
        micasense_image.set_point_calculator(sensor_model)
        return micasense_image
