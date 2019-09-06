from __future__ import division

from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.abstract_earth_overhead_point_calc \
    import AbstractEarthOverheadPointCalc
from resippy.photogrammetry import crs_defs
import gdal
import numpy as np
from numpy import ndarray


class RPCPointCalc(AbstractEarthOverheadPointCalc):
    """
    This is a concrete implementation of an AbstractEarthOverheadPointCalc.  It is an RPC Point Calculator, and
    computes pixel x / y locations using rational polynomials.
    """

    def __init__(self):
        self._samp_num_coeff = None      # type: ndarray
        self._samp_den_coeff = None       # type: ndarray
        self._samp_scale = None          # type: float
        self._samp_off = None            # type: float

        self._line_num_coeff = None      # type: ndarray
        self._line_den_coeff = None      # type: ndarray
        self._line_scale = None          # type: float
        self._line_off = None            # type: float

        self._lat_scale = None           # type: float
        self._lat_off = None             # type: float

        self._lon_scale = None           # type: float
        self._lon_off = None             # type: float

        self._height_scale = None        # type: float
        self._height_off = None          # type: float

        self.set_projection(crs_defs.PROJ_4326)
        self._bands_coregistered = True

    @classmethod
    def init_from_file(cls,
                       fname  # type: str
                       ):
        """
        This is a class method that initializes an RPC Point Calculator from a file.  The file is assumed to have
        RPC's (rational polynomial coefficients), which are readable and pare-able using GDAL.  It reads the
        coefficients from the file and uses 'init_from_coeffs' under the hood to initialize the point calculator.
        :param fname:
        :return:
        """
        dset = gdal.Open(fname)
        rpcs = dset.GetMetadata("RPC")
        dset = None

        samp_num_coeff = cls._str_to_numpy_arr(rpcs['SAMP_NUM_COEFF'])
        samp_den_coeff = cls._str_to_numpy_arr(rpcs['SAMP_DEN_COEFF'])
        samp_scale = float(rpcs['SAMP_SCALE'])
        samp_off = float(rpcs['SAMP_OFF'])

        line_num_coeff = cls._str_to_numpy_arr(rpcs['LINE_NUM_COEFF'])
        line_den_coeff = cls._str_to_numpy_arr(rpcs['LINE_DEN_COEFF'])
        line_scale = float(rpcs['LINE_SCALE'])
        line_off = float(rpcs['LINE_OFF'])

        lat_scale = float(rpcs['LAT_SCALE'])
        lat_off = float(rpcs['LAT_OFF'])

        long_scale = float(rpcs['LONG_SCALE'])
        long_off = float(rpcs['LONG_OFF'])

        height_scale = float(rpcs['HEIGHT_SCALE'])
        height_off = float(rpcs['HEIGHT_OFF'])

        return cls.init_from_coeffs(samp_num_coeff, samp_den_coeff, samp_scale, samp_off,
                                    line_num_coeff, line_den_coeff, line_scale, line_off,
                                    lat_scale, lat_off,
                                    long_scale, long_off,
                                    height_scale, height_off)

    @classmethod
    def init_from_coeffs(cls,
                         samp_num_coeff,        # type: ndarray
                         samp_den_coeff,         # type: ndarray
                         samp_scale,            # type: float
                         samp_off,              # type: float
                         line_num_coeff,        # type: ndarray
                         line_den_coeff,        # type: ndarray
                         line_scale,            # type: float
                         line_off,              # type: float
                         lat_scale,             # type: float
                         lat_off,               # type: float
                         lon_scale,             # type: float
                         lon_off,               # type: float
                         height_scale,          # type: float
                         height_off,            # type: float
                         ):
        """
        This is a class method that initializes an RPC Point Calculator from all of the required coefficients.
        :param samp_num_coeff: ndarray containing sample numerator coefficients
        :param samp_den_coeff: ndarray containing sample denominator coefficients
        :param samp_scale: float value the specifies sample scale factor
        :param samp_off: float value that specifies sample offset
        :param line_num_coeff: ndarray containing line numerator coefficients
        :param line_den_coeff: ndarray containing line denominator coefficients
        :param line_scale: float value that specifies line scale factor
        :param line_off: float value that specifies line offset
        :param lat_scale: float value that specifies latitude scale factor
        :param lat_off: float value that specifies latitude offset
        :param lon_scale: float value that specifies longitude scale factor
        :param lon_off: float value that specifies longitude offset
        :param height_scale: float value that specifies height scale factor
        :param height_off: float value that specifies height offset
        :return: RPC Point Calculator object
        """
        point_calc = cls()
        point_calc._samp_num_coeff = samp_num_coeff
        point_calc._samp_den_coeff = samp_den_coeff
        point_calc._samp_scale = samp_scale
        point_calc._samp_off = samp_off
        point_calc._line_num_coeff = line_num_coeff
        point_calc._line_den_coeff = line_den_coeff
        point_calc._line_scale = line_scale
        point_calc._line_off = line_off
        point_calc._lat_scale = lat_scale
        point_calc._lat_off = lat_off
        point_calc._lon_scale = lon_scale
        point_calc._lon_off = lon_off
        point_calc._height_scale = height_scale
        point_calc._height_off = height_off
        point_calc.set_approximate_lon_lat_center(lon_off, lat_off)

        return point_calc

    @classmethod
    def compute_p(cls,
                  coeffs,       # type: ndarray
                  x,            # type: ndarray
                  y,            # type: ndarray
                  z             # type: ndarray
                  ):
        """
        This method computes sample and line numerator and denominators from their respective coefficients
        :param coeffs: coefficients for line/sample numerator/denominator to be calculated
        :param x: normalized latitudes, calculated by (lats - lat_off)/lat_scale
        :param y: normalized longitudes, calculated by (lons - lon_off)/lon_scale
        :param z: normalized altitudes, calculated by (alts - height_off)/height_scale
        :return: the numerator or denominator for line or sample to be calculated given the provided coefficients
        """
        a1 = coeffs[0]
        a2 = coeffs[1]
        a3 = coeffs[2]
        a4 = coeffs[3]
        a5 = coeffs[4]
        a6 = coeffs[5]
        a7 = coeffs[6]
        a8 = coeffs[7]
        a9 = coeffs[8]
        a10 = coeffs[9]
        a11 = coeffs[10]
        a12 = coeffs[11]
        a13 = coeffs[12]
        a14 = coeffs[13]
        a15 = coeffs[14]
        a16 = coeffs[15]
        a17 = coeffs[16]
        a18 = coeffs[17]
        a19 = coeffs[18]
        a20 = coeffs[19]

        x_squared = x*x
        y_squared = y*y
        z_squared = z*z

        x_cubed = x_squared*x
        y_cubed = y_squared*y
        z_cubed = z_squared*z

        xy = x*y
        xz = x*z

        yz = y*z

        p = a1 + a2*y + a3*x + a4*z + a5*xy + a6*yz + a7*xz + a8 * y_squared + \
            a9*x_squared + a10*z_squared + a11*x*y*z + a12*y_cubed + a13*y*x_squared + a14*y*z_squared + \
            a15*y_squared*x + a16*x_cubed + a17*x*z_squared + a18*y_squared*z + a19*x_squared*y + a20*z_cubed
        return p

    def _lon_lat_alt_to_pixel_x_y_native(self,
                                         lons,  # type: ndarray
                                         lats,  # type: ndarray
                                         alts=None,  # type: ndarray
                                         band=None  # type: int
                                         ):  # type: (...) -> (ndarray, ndarray)
        """
        See documentation in AbstractEarthOverheadPointCalc
        :param lons:
        :param lats:
        :param alts:
        :param band:
        :return:
        """

        X_normalized = (lats - self._lat_off)/self._lat_scale
        Y_normalized = (lons - self._lon_off)/self._lon_scale
        Z_normalized = (alts - self._height_off)/self._height_scale

        p1 = self.compute_p(self._samp_num_coeff, X_normalized, Y_normalized, Z_normalized)
        p2 = self.compute_p(self._samp_den_coeff, X_normalized, Y_normalized, Z_normalized)
        p3 = self.compute_p(self._line_num_coeff, X_normalized, Y_normalized, Z_normalized)
        p4 = self.compute_p(self._line_den_coeff, X_normalized, Y_normalized, Z_normalized)

        x_norm = p1/p2
        y_norm = p3/p4

        x = x_norm * self._samp_scale + self._samp_off + 0.5
        y = y_norm * self._line_scale + self._line_off + 0.5

        return x, y

    def _pixel_x_y_alt_to_lon_lat_native(self,
                                         x_pixels,  # type: ndarray
                                         y_pixels,  # type: ndarray
                                         alts=None,  # type: ndarray
                                         band=None  # type: ndarray
                                         ):  # type: (...) -> (ndarray, ndarray)
        """
        See documentation in AbstractEarthOverheadPointCalc
        :param x_pixels:
        :param y_pixels:
        :param alts:
        :param band:
        :return:
        """
        return None

    @staticmethod
    def _str_to_numpy_arr(string_entry,     # type: str
                          ):
        """
        Converts string values to numpy arrays.  This is used to convert coefficients returned in a string
        format by GDAL.
        :param string_entry:
        :return: numpy array version of the a coefficient provided in string format
        """
        return np.array([float(x) for x in string_entry.split()])
