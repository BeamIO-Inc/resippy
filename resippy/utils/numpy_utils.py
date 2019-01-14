import numpy as np
from numpy import ndarray


def ndarray_zero_to_one(length  # type: int
                        ):  # type: (...) -> ndarray
    return np.arange(0, length) / (length - 1)


def ndarray_n_to_m(n,  # type: float
                   m,  # type: float
                   length  # type: int
                   ):  # type: (...) -> ndarray
    return ndarray_zero_to_one(length) * (m - n) + n
