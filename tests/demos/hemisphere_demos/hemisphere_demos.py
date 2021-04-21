import os
import numpy
from resippy.atmospheric_compensation.hemisphere_quads_model import HemisphereQuadsModel
from resippy.utils.image_utils import image_utils
from resippy.utils import file_utils
import matplotlib.pyplot as plt
from resippy.utils import video_utils
import imageio
import json


def create_hemisphere_quads_by_equal_spacings():
    n_azimuths = 20
    n_elevations = 10
    max_elevation_degrees = 80
    hemisphere = HemisphereQuadsModel.create_with_equal_az_el_spacings(n_azimuths, n_elevations, max_elevation_degrees)
    return hemisphere


def create_hemisphere_quads_by_equal_areas():
    n_azimuths = 20
    n_elevations = 10
    max_elevation_degrees = 80
    hemisphere = HemisphereQuadsModel.create_with_equal_areas(n_azimuths, n_elevations, max_elevation_degrees)
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


def quad_center_az_els_to_csv(n_azimuths,
                              n_elevations,
                              max_elevation,
                              output_csv_fname,
                              equal_area=True,
                              units='radians'):
    if equal_area:
        hemisphere = HemisphereQuadsModel.create_with_equal_areas(n_azimuths, n_elevations, max_elevation)
    else:
        hemisphere = HemisphereQuadsModel.create_with_equal_az_el_spacings(n_azimuths, n_elevations, max_elevation)
    hemisphere.quad_center_az_els_to_csv(output_csv_fname, units=units)


def animate_quad_colorization_by_spectral_band():
    spectral_json_file =  os.path.expanduser("~/Data/SMART/hemisphere_spectral_data/skydome_rad.json")
    output_dir = os.path.expanduser("~/Downloads/hemisphere_spectral_animation")
    file_utils.make_dir_if_not_exists(output_dir)
    with open(spectral_json_file) as f:
        data = json.load(f)
    n_azimuths = 36
    n_zeniths = int(len(data)/n_azimuths)
    hemisphere = HemisphereQuadsModel.create_with_equal_az_el_spacings(n_azimuths, n_zeniths, 81)
    counter = 0
    for elevation_index in reversed(range(n_zeniths)):
        for azimuth_index in range(n_azimuths):
            spectrum = numpy.array(data[counter]['total_rad'])
            hemisphere.set_quad_radiance(azimuth_index, elevation_index, spectrum)
            counter+=1
    hemisphere.initialize_uv_grayscale_image()

    for i in range(len(spectrum)):
        spectral_bandpass = numpy.zeros_like(spectrum)
        spectral_bandpass[i] = 1
        hemisphere.create_quad_radiance_uv_image(spectral_bandpass)
        colormapped_hemisphere = image_utils.apply_colormap_to_grayscale_image(hemisphere._uv_grayscale_image)
        output_fullpath = os.path.join(output_dir, "image" + str(i).zfill(10) + ".png")
        imageio.imwrite(output_fullpath, colormapped_hemisphere)
        print(i)


def save_spectrum_animation_movie():
    spectral_json_file =  os.path.expanduser("~/Data/SMART/hemisphere_spectral_data/skydome_rad.json")
    output_dir = os.path.expanduser("~/Downloads/spectral_animation_plots")
    file_utils.make_dir_if_not_exists(output_dir)
    raw_data = []
    with open(spectral_json_file) as f:
        data = json.load(f)
    for i in range(360):
        raw_data.append(data[i]['total_rad'])
    raw_data = numpy.array(raw_data)

    for i in range(raw_data.shape[1]):
        plt.clf()
        plt.plot(raw_data.transpose())
        plt.vlines(i, 0, raw_data.max())
        output_fname = os.path.join(output_dir, "plot_" + str(i).zfill(10) + ".png")
        plt.savefig(output_fname)

def combine_spectral_plots_and_hemispheres():
    spectral_plots_dir = os.path.expanduser("~/Downloads/spectral_animation_plots")
    hemisphere_animation_dir = os.path.expanduser("~/Downloads/hemisphere_spectral_animation")

    output_dir = os.path.expanduser("~/Downloads/combined_hemisphere_spectral_plot_images")
    file_utils.make_dir_if_not_exists(output_dir)

    spectral_fnames = sorted(file_utils.get_all_files_in_dir(spectral_plots_dir))
    hemisphere_fnames = sorted(file_utils.get_all_files_in_dir(hemisphere_animation_dir))

    hemisphere_image = imageio.imread(hemisphere_fnames[0])
    spectral_image = imageio.imread(spectral_fnames[0])

    new_image = numpy.zeros((hemisphere_image.shape[0], hemisphere_image.shape[1] + spectral_image.shape[1], 3))
    for i in range(1130):
        hemisphere_image = imageio.imread(hemisphere_fnames[i])
        spectral_image = imageio.imread(spectral_fnames[i])

        new_image[:, 0:hemisphere_image.shape[1], : ] = hemisphere_image
        new_image[0:spectral_image.shape[0] , hemisphere_image.shape[1]:hemisphere_image.shape[1] + spectral_image.shape[1], :] = spectral_image[:, :, 0:3]

        output_fullpath = os.path.join(output_dir, "image_" + str(i).zfill(10) + ".png")
        imageio.imwrite(output_fullpath, new_image)

def main():
    # visualize_hemisphere()
    # output_fname = os.path.expanduser("~/Downloads/hemisphere_az_els.csv")
    # quad_center_az_els_to_csv(12, 10, 80, output_fname, equal_area=True, units='degrees')

    # animate_quad_colorization_by_spectral_band()
    # images_dir = os.path.expanduser("~/Downloads/hemisphere_spectral_animation")
    # output_video_fname = os.path.expanduser("~/Downloads/hemisphere_spectral_vid.mp4")
    # video_utils.directory_of_images_to_mp4(images_dir, 60, output_video_fname)

    # save_spectrum_animation_movie()
    # combine_spectral_plots_and_hemispheres()
    images_dir = os.path.expanduser("~/Downloads/combined_hemisphere_spectral_plot_images")
    output_video_fname = os.path.expanduser("~/Downloads/hemisphere_spectral_vid_w_plot.gif")
    image_fnames = sorted(file_utils.get_all_files_in_dir(images_dir))
    images_to_use = image_fnames[0:800:8]
    video_utils.list_of_fnames_to_animated_gif(images_to_use, 0.1, output_video_fname)


if __name__ == '__main__':
    main()
