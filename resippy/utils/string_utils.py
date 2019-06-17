import os


def cleanup_newlines(text   # type: str
                     ):     # type: (...) -> str
    """
    This is a convenience method that finds newline characters specified by \r, \n, or \r\n and replaces them
    with newline characters specified by os.linesep.
    :param text: input string
    :return: output string, same as input but with newline characters replaced in a standard way using os.linesep
    """
    newline_chars = ['\r\n', '\r', '\n']
    for newline_char in newline_chars:
        text = text.replace(newline_char, os.linesep)
    return text


def remove_newlines(text   # type: str
                    ):     # type: (...) -> str
    """
    This is a convenience method that removes newlines from a string.  It will call 'cleanup_newlines' first to ensure
    all the newline characters are in a standard format so they can be found.
    :param text: input string
    :return: output string, same as input but with newline characters removed.
    """
    clean_newlines_text = cleanup_newlines(text)
    return clean_newlines_text.replace(os.linesep, '')

