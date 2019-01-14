import os


def cleanup_newlines(text   # type: str
                     ):     # type: (...) -> str
    newline_chars = ['\r\n', '\r', '\n']
    for newline_char in newline_chars:
        text = text.replace(newline_char, os.linesep)
    return text


def remove_newlines(text   # type: str
                    ):     # type: (...) -> str
    clean_newlines_text = cleanup_newlines(text)
    return clean_newlines_text.replace(os.linesep, '')

