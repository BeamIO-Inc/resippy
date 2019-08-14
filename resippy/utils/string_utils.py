import os
from re import sub


def cleanup_newlines(text,   # type: str
                     ):     # type: (...) -> str
    """
    stuff
    """

    newline_chars = ['\r\n', '\r', '\n']
    for newline_char in newline_chars:
        text = text.replace(newline_char, os.linesep)
    return text


def remove_newlines(text,   # type: str
                    ):     # type: (...) -> str
    """
    This is a convenience method that removes newlines from a string.  It will call 'cleanup_newlines' first to ensure
    all the newline characters are in a standard format so they can be found.
    :param text: input string
    :return: output string, same as input but with newline characters removed.

    """

    clean_newlines_text = cleanup_newlines(text)
    return clean_newlines_text.replace(os.linesep, '')


def convert_to_snake_case(text     # type: str
                           ):       # type: (...) -> str
    """
    This is a convenience method that converts a camel case string to snake case.
    :param text: input string
    :return: output string, same as input except snake case
    """
    s1 = sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

