import osr
from pyproj import Proj
import utm


def wkt_to_proj_string(wkt,            # type: str
                ):              # type: (...) -> Proj
    spatialref = osr.SpatialReference()
    spatialref.ImportFromWkt(wkt)
    proj_string = spatialref.ExportToProj4()
    return proj_string


def wkt_to_proj(wkt,            # type: str
                ):              # type: (...) -> Proj
    proj_string = wkt_to_proj_string(wkt)
    proj = Proj(proj_string)
    return proj


def wkt_to_osr_spatialref(wkt,  # type: str
                          ):              # type: (...) -> Proj
    spatialref = osr.SpatialReference()
    spatialref.ImportFromWkt(wkt)
    return spatialref


def decimal_degrees_to_local_utm_proj(lon_decimal_degrees,      # type: float
                                      lat_decimal_degrees,      # type: float
                                      ):                        # type: (...) -> Proj
    zone_info = utm.from_latlon(lat_decimal_degrees, lon_decimal_degrees)
    zone_string = str(zone_info[2]) + zone_info[3]
    zone_letter = zone_string[-1]
    is_south_zone = zone_letter < 'N'
    local_proj = Proj(proj='utm', ellps='WGS84', zone=zone_string, south=is_south_zone)
    return local_proj
