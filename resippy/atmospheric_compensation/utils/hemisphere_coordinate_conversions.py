import numpy


def az_el_to_uv_coords(azimuths,  # type: ndarray
                       elevations,  # type: ndarray
                       ):
    u0 = numpy.cos(azimuths)
    v0 = numpy.sin(azimuths)
    u = u0 * (numpy.pi / 2 - elevations) / (numpy.pi / 2)
    v = v0 * (numpy.pi / 2 - elevations) / (numpy.pi / 2)
    u = (1 + u) / 2.0
    v = (1 + v) / 2.0
    return u, v


def uv_coords_to_az_el_coords(u,  # type: ndarray
                              v,  # type: ndarray
                              ):
    u1 = 2 * u - 1
    v1 = 2 * v - 1
    az = numpy.arctan2(v1, u1)
    az[numpy.where(az < 0)] = az[numpy.where(az < 0)] + 2 * numpy.pi
    el = numpy.pi / 2.0 * (1 - u1 / numpy.cos(az))
    el[numpy.where(el < 0)] = 0
    return az, el


def uv_coords_to_uv_pixel_yx_coords(uv_npixels,  # type: int
                                    u_coords,  # type: ndarray
                                    v_coords,  # type: ndarray
                                    ):
    nx = uv_npixels
    ny = uv_npixels
    x = u_coords * nx - 0.5
    y = ((1 - v_coords) * ny) - 0.5
    return y, x


def uv_pixel_yx_coords_to_uv_coords(uv_npixels,  # type: int
                                    y_pixels,  # type: ndarray
                                    x_pixels,  # type: ndarray
                                    ):
    u = (x_pixels + 0.5) / uv_npixels
    v = (uv_npixels - y_pixels - 0.5) / uv_npixels
    return u, v


def uv_pixel_yx_coords_to_az_el(n_uv_pixels,  # type: int
                                pixels_y,  # type: ndarray
                                pixels_x,  # type: ndarray
                                ):
    u, v = uv_pixel_yx_coords_to_uv_coords(n_uv_pixels, pixels_y, pixels_x)
    return uv_coords_to_az_el_coords(u, v)


def az_el_to_uv_pixel_yx_coords(uv_npixels,
                                azimuths,
                                elevations,
                                ):
    uv_coords = az_el_to_uv_coords(azimuths, elevations)
    return uv_coords_to_uv_pixel_yx_coords(uv_npixels, uv_coords[0], uv_coords[1])
