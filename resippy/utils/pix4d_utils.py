from __future__ import division

import numpy as np
from numpy import ndarray
from resippy.utils import string_utils
import os
from pyproj import Proj
import osgeo.osr as osr


def read_p_matrices(pmatrix_txt_file    # type: str
                    ):                  # type: (...) -> dict
    # TODO: create working example for this or remove, it doesn't seem to be used by any other methods in the project.
    p_matricies = {}
    with open(pmatrix_txt_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line_no_linesep = string_utils.remove_newlines(line.strip())
        line_fields = line_no_linesep.split(' ')
        params = np.asarray(line_fields[1:], dtype=float)
        fname = line_fields[0]
        p_matricies[fname] = params
    return p_matricies


def read_calibrated_camera_parameters(calibrated_params_file    # type: str
                                      ):                        # type: (...) -> dict
    """
    This method reads the calibrated camera parameters for an entire pix4d collect and returns a dictionary
    The dictionary keys are the base image file name.  The values are another dictionary that have the following
    key / value pairs:
    'camera_matrix_k': 3x3 numpy array containing the camera k matrix
    'camera_position_t': ndarray of length 3 containing the camera position 't'
    'camera_rotation_r': 3x3 ndarray containing the camera's rotation matrix
    'n_x_pixels': number of x pixels
    'n_y_pixels': number of y pixels
    'radial_distortion': numpy array containing 3 radial distortion parameters
    'tangential_distortion': numpy array containing 2 tangential distortion parameters.
    :param calibrated_params_file:
    :return: dictionary of image filenames and calibrated parameters
    """
    calibrated_params = {}
    with open(calibrated_params_file, 'r') as f:
        lines = f.readlines()
    cleaned_up_lines = [string_utils.remove_newlines(x) for x in lines]
    params = cleaned_up_lines[8:]
    per_file_params = [params[x:x + 10] for x in range(0, len(params), 10)]
    for file_params in per_file_params:
        fname, image_w, image_h = file_params[0].split(" ")
        tmp_list = np.array(file_params[1:4])
        camera_matrix_k = np.array([list(map(float, row.split())) for row in tmp_list])
        tmp_list = np.array(file_params[4:5])
        radial_distortion = np.array(np.squeeze([list(map(float, row.split())) for row in tmp_list]))
        tmp_list = np.array(file_params[5:6])
        tangential_distortion = np.array(np.squeeze([list(map(float, row.split())) for row in tmp_list]))
        tmp_list = np.array(file_params[6:7])
        camera_position_t = np.array([list(map(float, row.split())) for row in tmp_list])
        tmp_list = np.array(file_params[7:10])
        camera_rotation_r = np.array([list(map(float, row.split())) for row in tmp_list])
        fname_params = {"n_x_pixels": int(image_w),
                        "n_y_pixels": int(image_h),
                        "camera_matrix_k": camera_matrix_k,
                        "radial_distortion": radial_distortion,
                        "tangential_distortion": tangential_distortion,
                        "camera_position_t": camera_position_t,
                        "camera_rotation_r": camera_rotation_r}
        calibrated_params[fname] = fname_params
    return calibrated_params


def read_calibrated_external_camera_parameters(calibrated_external_params_file  # type: str
                                               ):                               # type: (...) -> dict
    """
    Reads external orientation parameters from a file for an entire pix4d collect
    :param calibrated_external_params_file: external orientation parameters file to read
    :return: dictionary of external orientation parameters.
    Key is the base image file name
    value is a numpy array of length 6 containing x, y, z, omega, phi, kappa
    Units are specified by the collect's projection (x, y, z), and degrees for omega, phi, kappa (roll, pitch, yaw)
    """
    external_params = {}
    with open(calibrated_external_params_file, 'r') as f:
        lines = f.readlines()
    cleaned_up_lines = [string_utils.remove_newlines(x) for x in lines][1:]
    for file_params in cleaned_up_lines:
        fname, x, y, z, omega, phi, kappa = file_params.split(" ")
        external_params[fname] = [float(x), float(y), float(z), float(omega), float(phi), float(kappa)]
    return external_params


def create_r_matrix(external_parameters     # type: list
                    ):                      # type: (...) -> ndarray
    """
    Creates a rotation matrix from camera's external orientation parameters
    :param external_parameters: numpy array of length 6 (x, y, z, omega, phi kappa) Where:
    omega = roll, in degrees
    phi = pitch, in degrees
    kappa = roll, in degrees
    :return: 3x3 rotation matrix, as a numpy array
    """
    omega, phi, kappa = np.deg2rad(external_parameters[3]), np.deg2rad(external_parameters[4]), \
                        np.deg2rad(external_parameters)[5]
    r_11 = np.cos(kappa) * np.cos(phi)
    r_12 = -np.sin(kappa) * np.cos(phi)
    r_13 = np.sin(phi)

    r_21 = np.cos(kappa) * np.sin(omega) * np.sin(phi) + np.sin(kappa) * np.cos(omega)
    r_22 = np.cos(kappa) * np.cos(omega) - np.sin(kappa) * np.sin(omega) * np.sin(phi)
    r_23 = -np.sin(omega) * np.cos(phi)

    r_31 = np.sin(kappa) * np.sin(omega) - np.cos(kappa) * np.cos(omega) * np.sin(phi)
    r_32 = np.sin(kappa) * np.cos(omega) * np.sin(phi) + np.cos(kappa) * np.sin(omega)
    r_33 = np.cos(omega) * np.cos(phi)

    r = [[r_11, r_12, r_13],
         [r_21, r_22, r_23],
         [r_31, r_32, r_33]]

    return np.array(r)


def make_master_dict(base_path,                     # type: str
                     external_params_fname=None,    # type: str
                     internal_params_fname=None,    # type: str
                     calibrated_params_fname=None,  # type: str
                     wkt_fname=None,                # type: str
                     ):                             # type: (...) -> dict
    """
    This method takes files generated by pix4d and creates a master dictionary of parameters for all camera files
    generated during a collect.  The dictionary keys are the base image filename.  The values are another dictionary
    containing all of the parameters.  The parameter dictionary has the following keys and values:
    'camera_matrix_k': a 3x3 numpy array of the camera k matrix
    'camera_position_t': a numpy array containing the camera position 't'
    'camera_rotation_r': 3x3 numpy array containing the camera rotation matrix
    'external_params': a numpy array of length 6 containing camera x, y, z, in units specified by the collect's projection
    and omega, phi, kappa (roll, pitch, yaw) values, in degrees
    'internal_params': a dictionary of internal camera orientation parameters.  The format is the same as the
    dictionaries generated by 'read_pix4d_calibrated_internal_camera_parameters'
    'n_x_pixels': number of x pixels for the given camera
    'n_y_pixels': number of y pixels for the given camera
    'radial_distortion': numpy array containing 3 radial distortion parameters
    'tangential_distortion': numpy array containing 2 tangential distortion parameters
    :param base_path: base path of the collect.  This method will attempt to find all of the files for the collect
    starting at the base path if this is the only parameter specified.
    :param external_params_fname: external parameters file, defaults to None
    :param internal_params_fname: internal parameters file, defaults to None
    :param calibrated_params_fname: calibrated parameters file, defaults to None
    :param wkt_fname: wkt filename, defaults to None
    :return:
    """

    default_external_params_fname_text = "calibrated_external_camera_parameters.txt"
    default_internal_params_fname_text = "pix4d_calibrated_internal_camera_parameters.cam"
    default_params_fname_text = "calibrated_camera_parameters.txt"
    default_wkt_fname_text = "wkt.prj"

    files_in_base_path = os.listdir(base_path)

    if external_params_fname is None:
        external_params_fname = [s for s in files_in_base_path if s.endswith(default_external_params_fname_text)][0]

    if internal_params_fname is None:
        internal_params_fname = [s for s in files_in_base_path if s.endswith(default_internal_params_fname_text)][0]

    if calibrated_params_fname is None:
        calibrated_params_fname = [s for s in files_in_base_path if s.endswith(default_params_fname_text)][0]

    if wkt_fname is None:
        wkt_fname = [s for s in files_in_base_path if s.endswith(default_wkt_fname_text)][0]

    external_params_path = os.path.join(base_path, external_params_fname)
    internal_params_path = os.path.join(base_path, internal_params_fname)
    calibrated_params_path = os.path.join(base_path, calibrated_params_fname)
    wkt_path = os.path.join(base_path, wkt_fname)

    external_dict = read_calibrated_external_camera_parameters(external_params_path)
    internal_list = read_pix4d_calibrated_internal_camera_parameters(internal_params_path)
    calibrated_dict = read_calibrated_camera_parameters(calibrated_params_path)

    master_dict = {}

    # set master dict projection
    proj_wkt = open(wkt_path, "r").read()
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(proj_wkt)
    proj4_string = spatial_ref.ExportToProj4()
    projection = Proj(proj4_string)
    master_dict['projection'] = projection

    for key in calibrated_dict.keys():
        entry_dict = {}
        calibrated_entries = calibrated_dict[key]
        for entry in calibrated_entries:
            entry_dict[entry] = calibrated_entries[entry]
        external_list = external_dict[key]
        entry_dict["external_params"] = external_list
        cam_index_str = os.path.splitext(key)[0].split("_")[-1]
        if len(cam_index_str) != 1:
            raise ValueError("camera index string has more than one character!")
        cam_index = int(cam_index_str) - 1
        entry_dict["internal_params"] = internal_list[cam_index]
        master_dict[key] = entry_dict
        master_dict["r_matrix_computed"] = create_r_matrix(external_dict[key])
    return master_dict


def read_pix4d_calibrated_internal_camera_parameters(internal_params_cam_file,  # type: str
                                                     search_text="Pix4D",       # type: str
                                                     ):                         # type: (...) -> list
    """
    Reads the internal parameters camera file generated by pix4d and return a list of dictionaries, which contain
    values for each camera's internal parameters accordin to the Pix4d documentation.
    For each camera, the following parameters are contained in the dictionary:
    'F': focal length in millimeters
    'K1': Radial distortion parameter 1
    'K2': Radial distortion parameter 2
    'K3': Radial distortion parameter 3
    'Px': Principal point  in the x pixel direction
    'Py': Principal point in the y pixel direction
    'T1': Tangential distortion parameter 1
    'T2': Tangential distortion parameter 2
    'fpa_height_mm': focal plane height, in millimeters
    'fpa_width_mm': focal plane width, in millimeters
    :param internal_params_cam_file:
    :param search_text:
    :return: list of dictionaries containing the internal camera parameters
    """
    with open(internal_params_cam_file, 'r') as f:
        lines = f.readlines()
    lines = [string_utils.remove_newlines(x) for x in lines]
    new_camera_param_start_locations = []

    for i, line in enumerate(lines):
        if search_text in line:
            new_camera_param_start_locations.append(i)

    new_camera_param_end_locations = new_camera_param_start_locations[1:]
    new_camera_param_end_locations.append(i)

    camera_params_list = []

    for i in range(len(new_camera_param_start_locations)):
        internal_params_dict = {}
        text_block = lines[new_camera_param_start_locations[i] + 1:new_camera_param_end_locations[i]]
        text_block = list(filter(lambda a: a != '', text_block))
        for txt in text_block:
            if txt.startswith("#"):
                if "Focal Length mm" in txt:
                    fpa_width_mm, fpa_height_mm = txt[txt.rfind(" ") + 1:].replace("mm", "").split("x")
                    internal_params_dict["fpa_width_mm"] = float(fpa_width_mm)
                    internal_params_dict["fpa_height_mm"] = float(fpa_height_mm)
                else:
                    pass
            else:
                key, val = txt.split(" ")
                internal_params_dict[key] = float(val)
        camera_params_list.append(internal_params_dict)
    return camera_params_list
