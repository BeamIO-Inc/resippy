import numpy as np
from numpy import ndarray
from typing import Any


def ndarray_zero_to_one(length  # type: int
                        ):  # type: (...) -> ndarray
    return np.arange(0, length) / (length - 1)


def ndarray_n_to_m(n,  # type: float
                   m,  # type: float
                   length  # type: int
                   ):  # type: (...) -> ndarray
    return ndarray_zero_to_one(length) * (m - n) + n


def list_of_tuples_to_separate_lists(tuples,        # type: list
                                     ):             # type: (...) -> list
    return np.array(tuples).transpose().tolist()


def separate_lists_to_list_of_tuples(lists,         # type: list
                                     ):             # type: (...) -> list
    list_of_tuples = []
    for entry in np.transpose(lists):
        list_of_tuples.append(tuple(entry))
    return list_of_tuples

def remove_all_entries_from_list(input_list,        # type: list
                                 value_to_remove    # type: Any
                                 ):                 # type: (...) -> list
    return [x for x in input_list if x != value_to_remove]