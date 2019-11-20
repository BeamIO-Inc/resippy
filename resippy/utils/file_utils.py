import os
from functools import reduce
from typing import Union
import glob
import resippy.utils.string_utils as string_utils


def make_dir_if_not_exists(dir_to_create    # type: str
                           ):               # type: (...) -> None
    if not os.path.exists(dir_to_create):
        os.makedirs(dir_to_create)


def get_all_folders_in_dir(directory        # type: str
                           ):               # type: (...) -> list
    return list(filter(os.path.isdir, [os.path.join(directory, item) for item in os.listdir(directory)]))


def get_all_files_in_dir(directory,                     # type: str
                         extensions=None,               # type: Union[str, list]
                         return_fullpaths=True          # type: bool
                         ):                             # type: (...) -> list
    sanitized_extensions = []
    if type(extensions) == (type([])):
        for extension in extensions:
            sanitized_extensions.append(extension.replace(".", "").replace("*", ""))
    elif type(extensions) == (type("")):
        sanitized_extensions.append(extensions.replace(".", "").replace("*", ""))
    elif extensions is None:
        sanitized_extensions.append("*")
    else:
        raise TypeError("extensions should either be a string or list of strings")

    files = []
    for extension in sanitized_extensions:
        fnames = glob.glob(directory + os.path.sep + "*." + extension)
        files = files + fnames
    if return_fullpaths:
        return files
    else:
        basenames = []
        for f in files:
            basenames.append(os.path.basename(f))
        return basenames


def get_files_in_dir_that_start_with(directory,                 # type: str
                                     starts_with,               # type: Union[str, list]
                                     extensions=None,           # type: Union[str, list]
                                     return_fullpaths=True      # type: bool
                                     ):                         # type: (...) -> list
    all_files = get_all_files_in_dir(directory, extensions)
    all_basenames = [os.path.basename(x) for x in all_files]
    if type(starts_with) == (type("")):
        starts_with = [starts_with]
    filtered_basenames = []
    for starts_with_text in starts_with:
        for basename in all_basenames:
            if basename.startswith(starts_with_text):
                filtered_basenames.append(basename)
    if not return_fullpaths:
        return filtered_basenames
    else:
        fullpaths = [os.path.join(directory, x) for x in filtered_basenames]
        return fullpaths


def get_path_from_subdirs(base_dir,     # type: str
                          subdirs       # type: list
                          ):            # type: (...) -> str
    paths_list = [base_dir]
    paths_list.extend(subdirs)
    abs_path = reduce(os.path.join, paths_list)
    return abs_path


def write_text_list_to_file(text_list,  # type: list
                            output_fname,  # type: str
                            ):              # type: (...) -> None
    text_list_with_newlines = [entry + '\n' for entry in text_list]
    text_list_with_newlines[-1] = text_list[-1]
    with open(output_fname, 'w') as f:
        f.writelines(text_list_with_newlines)


def read_text_list_from_file(text_file_fname,   # type: str
                             ):                 # type: (...) -> list
    with open(text_file_fname, 'r') as f:
        text_list = f.readlines()
    for i in range(len(text_list)):
        text_list[i] = string_utils.remove_newlines(text_list[i])
    return text_list
