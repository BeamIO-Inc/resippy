from resippy.utils import file_utils
from moviepy.video.io import ImageSequenceClip
import imageio



def directory_of_images_to_mp4(dir,  # type: str
                               fps,  # type: float
                               output_fname,  # type: str
                               ):
    image_files = sorted(file_utils.get_all_files_in_dir(dir))
    clip = ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(output_fname)

def directory_of_images_to_animated_gif(dir,
                                        duration_per_rame,
                                        output_fname):
    image_files = sorted(file_utils.get_all_files_in_dir(dir))
    with imageio.get_writer(output_fname, mode='I', duration=duration_per_rame) as writer:
        for filename in image_files:
            image = imageio.imread(filename)
            writer.append_data(image)

def list_of_fnames_to_animated_gif(fnames,
                                   duration_per_rame,
                                   output_fname):
    with imageio.get_writer(output_fname, mode='I', duration=duration_per_rame) as writer:
        for filename in fnames:
            image = imageio.imread(filename)
            writer.append_data(image)
