import os

from osgeo import gdal
import numpy as np


ENVI_DTYPES_TO_NUMPY = {"1": np.uint8,
                        "2": np.int16,
                        "3": np.int32,
                        "4": np.float32,
                        "5": np.float64,
                        "6": None,
                        "7": None,
                        "8": None,
                        "9": np.complex64,
                        "10": None,
                        "11": None,
                        "12": np.uint16,
                        "13": np.uint32,
                        "14": np.int64,
                        "15": np.uint64,
                        }


def envi_dtype_to_numpy_dtype(envi_dtype,       # type: int
                              ):
    return ENVI_DTYPES_TO_NUMPY[str(int(envi_dtype))]


def numpy_dtype_to_envi_dtype(envi_dtype,       # type: int
                              ):
    for key, val in ENVI_DTYPES_TO_NUMPY.items():
        if val == envi_dtype:
            return int(key)


def read_envi_header(envi_fname,        # type: str
                     ):                 # type:(...)-> dict
    im = gdal.Open(envi_fname)
    metadata = im.GetMetadata(domain='ENVI')
    return metadata


# TODO this is a hack to get a baseline writer working without relying on GDAL
def write_envi_header(header_dict,      # type: dict
                      header_fname,     # type: str
                     ):                 # type:(...)-> None
    with open(header_fname, "w") as f:
        f.write("ENVI" + "\n")
        for key, val in header_dict.items():
            f.write(key + " = " + str(val))
            f.write("\n")


# TODO: add more interleave formats, only BIP is currently supported
def write_envi_image(image_cube,                    # type: np.ndarray
                     output_fname,                  # type: str
                     output_header_fname=None,      # type: str
                     interleave='bil',              # type: str
                     ):                             # type: (...) -> None
    if output_header_fname is None:
        output_header_fname = output_fname + ".hdr"
    ny, nx, nbands = image_cube.shape
    raw_data = np.zeros(ny*nx*nbands, dtype=image_cube.dtype)
    if interleave == 'bil':
        raw_data_counter = 0
        for l in range(ny):
            for b in range(nbands):
                data_chunk = image_cube[l, :, b]
                raw_data[raw_data_counter:raw_data_counter+nx] = data_chunk
                raw_data_counter = raw_data_counter+nx
    raw_data.tofile(output_fname)

    envi_dtype = numpy_dtype_to_envi_dtype(image_cube.dtype)

    envi_header = {}
    envi_header['samples'] = int(nx)
    envi_header['lines'] = int(ny)
    envi_header['bands'] = int(nbands)
    envi_header['data type'] = int(envi_dtype)
    envi_header['interleave'] = interleave
    write_envi_header(envi_header, output_header_fname)


def read_envi_image(image_file,             # type: str
                    header_file=None,       # type: str
                    ):                      # type: (...) -> np.ndarray
    if header_file is None:
        try:
            header_guesses = [image_file[:-4] + ".hdr", image_file + ".hdr"]
            for header_guess in header_guesses:
                if os.path.exists(header_guess):
                    header_file = header_guess
        except FileNotFoundError:
            print("Can't find a corresponding header file for this ENVI image.")
    envi_header = read_envi_header(header_file)
    nx = int(envi_header['samples'])
    ny = int(envi_header['lines'])
    nbands = int(envi_header['bands'])
    envi_dtype = envi_header['data type']
    interleave = envi_header['interleave']
    envi_image_data = read_all_image_data_to_numpy_array(image_file, envi_dtype)
    image_cube = construct_image_cube_from_raw_data(envi_image_data, nx, ny, nbands, interleave)
    return image_cube


def read_all_image_data_to_numpy_array(envi_image_path,     # type: str
                                       envi_dtype,          # type: int
                                       ):                   # type: (...) -> np.ndarray
    numpy_dtype = envi_dtype_to_numpy_dtype(envi_dtype)
    numpy_data = np.fromfile(envi_image_path, numpy_dtype)
    return numpy_data


def construct_image_cube_from_raw_data(raw_envi_data,       # type: np.ndarray
                                       nx,                  # type: int
                                       ny,                  # type: int
                                       nbands,              # type: int
                                       interleave,          # type: str
                                       ):                   # type: (...) -> np.ndarray
    image_cube = np.zeros((ny, nx, nbands), dtype=raw_envi_data.dtype)
    if interleave == 'bil':
        raw_data_counter = 0
        for l in range(ny):
            for b in range(nbands):
                data_chunk = raw_envi_data[raw_data_counter:raw_data_counter+nx]
                image_cube[l, :, b] = data_chunk
                raw_data_counter = raw_data_counter+nx
    return image_cube
