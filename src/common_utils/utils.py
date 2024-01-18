import numpy as np


def generate_shifted_dirac(array: np.ndarray, shift: float) -> np.ndarray:
    dirac_signal = np.zeros_like(array)
    index = index_from_value(array=array, value=shift)
    dirac_signal[index] = 1
    return dirac_signal


def index_from_value(array: np.ndarray, value: float) -> int:
    index = (np.abs(array - value)).argmin()
    return index
