import numpy
from resippy.atmospheric_compensation.hemisphere_quads_model import HemisphereQuadsModel


def create_hemisphere_quads_by_equal_spacings():
    n_azimuths = 20
    n_elevations = 10
    max_elevation_degrees = 80
    hemisphere = HemisphereQuadsModel.create_from_equal_az_el_spacings(n_azimuths, n_elevations, max_elevation_degrees)
    return hemisphere


def create_hemisphere_quads_by_equal_areas():
    n_azimuths = 20
    n_elevations = 10
    max_elevation_degrees = 80
    hemisphere = HemisphereQuadsModel.create_from_equal_areas(n_azimuths, n_elevations, max_elevation_degrees)
    return hemisphere


def create_hemispheres():
    equal_spacing_hemisphere = create_hemisphere_quads_by_equal_spacings()
    equal_spacing_hemisphere.center_xyz = numpy.array([0, 0, 0])
    az_el_quads_degrees = equal_spacing_hemisphere.all_quad_az_els_degrees
    az_el_quads_radians = equal_spacing_hemisphere.all_quad_az_els_radians
    xyz_quads = equal_spacing_hemisphere.all_quad_xyzs
    cap_degrees = equal_spacing_hemisphere.cap_az_els_degrees
    cap_radians = equal_spacing_hemisphere.cap_polygon_radians
    cap_xyzs = equal_spacing_hemisphere.cap_xyz_polygon

    elevation_spacings = equal_spacing_hemisphere.equal_area_elevation_angles(3, 80, max_elevation_units='degrees')

    equal_area_hemisphere = create_hemisphere_quads_by_equal_areas()


def visualize_hemisphere():
    hemisphere = HemisphereQuadsModel.create_from_equal_areas(4, 5, 80)
    hemisphere.center_xyz = [0, 0, 0]
    quads = hemisphere.all_quad_xyzs
    hemisphere = hemisphere.create_trimesh_model()


def main():
    visualize_hemisphere()


if __name__ == '__main__':
    main()
