from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.rpc_point_calc import RPCPointCalc
from resippy.image_objects.earth_overhead.earth_overhead_point_calculators.pinhole_camera import PinholeCamera

from resippy.utils import photogrammetry_utils
from resippy.utils.image_utils import image_utils
import numpy as np
import time


def rpc_timings():
    samp_num_coeff = [-2.401488e-03,  1.014755e+00,  1.773499e-02,  2.048626e-02, -4.609470e-05,
                      4.830748e-04, -2.015272e-04,  1.212827e-03,  5.065720e-06,  3.740396e-05,
                      -1.582743e-07,  1.437278e-06,  3.620892e-08,  2.144755e-07, -1.333671e-07,
                      0.000000e+00, -5.229308e-08,  1.111695e-06, -1.337535e-07,  0.000000e+00]

    samp_den_coeff = [1.000000e+00,  1.197118e-03,  9.340466e-05, -4.381989e-04,  3.359669e-08,
                      0.000000e+00,  2.959469e-08,  1.412447e-06,  8.398708e-08, -1.782544e-07,
                      0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
                      0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00]
    samp_scale = 1283.0
    samp_off = 1009.0
    height_off = 42.0
    height_scale = 501.0
    lat_off = 32.8902
    lat_scale = 0.0142

    line_num_coeff = [3.448567e-03,  1.975650e-02, -1.147937e+00,  1.622923e-01,  5.249710e-05,
                      4.231537e-05,  3.559660e-04, -9.052539e-05, -1.932047e-03, -2.341649e-05,
                      -1.025999e-06,  0.000000e+00, -1.116629e-07,  1.635365e-07, -1.704056e-07,
                      -3.139215e-06,  1.607936e-06,  1.281797e-07,  1.412060e-06, -2.638793e-07]

    line_den_coeff = [1.000000e+00, -1.294475e-05, 1.682046e-03, -1.481430e-04, 1.503771e-07,
                      7.529185e-07, -4.596928e-07, 0.000000e+00, 2.736755e-06, -1.609702e-06,
                      3.466453e-08, 1.066716e-08, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                      0.000000e+00, -7.020142e-08, 1.766701e-08, 2.536018e-07, 0.000000e+00]

    line_off = 856.0
    line_scale = 1143.0
    lon_off = 13.1706
    lon_scale = 0.0167

    point_calc = RPCPointCalc.init_from_coeffs(samp_num_coeff, samp_den_coeff, samp_scale, samp_off,
                                               line_num_coeff, line_den_coeff, line_scale, line_off,
                                               lat_scale, lat_off, lon_scale, lon_off, height_scale, height_off)

    lon_center = lon_off
    lat_center = lat_off
    d_lon = 0.001
    d_lat = 0.001

    nx = 2000
    ny = 2000

    ground_grid = photogrammetry_utils.create_ground_grid(lon_center - d_lon, lon_center + d_lon,
                                                          lat_center - d_lat, lat_center + d_lat,
                                                          nx, ny)

    lons = image_utils.flatten_image_band(ground_grid[0])
    lats = image_utils.flatten_image_band(ground_grid[1])
    alts = np.zeros_like(lats)

    n_loops = 4

    tic = time.time()
    for n in range(n_loops):
        point_calc.compute_p(samp_num_coeff, lats, lons, alts)
    toc = time.time()
    print("calculated " + str(n_loops*nx*ny) + " pixels in " + str(toc-tic) + " seconds.")
    print(str(n_loops*nx*ny/(toc-tic)/1e6) + " Megapixels per second")


def pinhole_timings():

    point_calc = PinholeCamera()
    point_calc.init_pinhole_from_coeffs(0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 50.0)

    lon_center = 0
    lat_center = 0
    d_lon = 500
    d_lat = 500

    nx = 2000
    ny = 2000

    ground_grid = photogrammetry_utils.create_ground_grid(lon_center - d_lon, lon_center + d_lon,
                                                          lat_center - d_lat, lat_center + d_lat,
                                                          nx, ny)

    lons = image_utils.flatten_image_band(ground_grid[0])
    lats = image_utils.flatten_image_band(ground_grid[1])
    alts = np.zeros_like(lats)

    n_loops = 4

    tic = time.time()
    for n in range(n_loops):
        point_calc.world_to_image_plane(lons, lats, alts)
    toc = time.time()
    print("calculated " + str(n_loops*nx*ny) + " pixels in " + str(toc-tic) + " seconds.")
    print(str(n_loops*nx*ny/(toc-tic)/1e6) + " Megapixels per second")

def main():
    rpc_timings()
    pinhole_timings()


if __name__ == "__main__":
    main()

