from resippy.utils import file_utils
from moviepy.video.io import ImageSequenceClip


def directory_of_images_to_mp4(dir,  # type: str
                               fps,  # type: float
                               output_fname,  # type: str
                               ):
    image_files = sorted(file_utils.get_all_files_in_dir(dir))
    clip = ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(output_fname)
