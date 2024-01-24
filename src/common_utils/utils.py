import numpy as np

from src.common_utils.custom_vars import Opd, Wvn


def calculate_phase_difference(
        opds: np.ndarray[tuple[Opd], np.dtype[np.float_]],
        wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
) -> np.ndarray[tuple[Opd, Wvn], np.dtype[np.float_]]:
    return opds[:, None] * wavenumbers[None, :]


def generate_shifted_dirac(array: np.ndarray, shift: float) -> np.ndarray:
    dirac_signal = np.zeros_like(array)
    index = index_from_value(array=array, value=shift)
    dirac_signal[index] = 1
    return dirac_signal


def index_from_value(array: np.ndarray, value: float) -> int:
    index = (np.abs(array - value)).argmin()
    return index


def generate_sampled_opds(nb_opd: int, del_opd: float):
    opds = del_opd * np.arange(nb_opd)
    return opds


def generate_wavenumbers_from_opds(nb_wn: int, del_opd: float):
    del_wn = 1 / (2 * nb_wn * del_opd)  # del_wn tends to zero as nb_wn tends to infinity (implies continuous)
    wavenumbers = del_wn * (np.arange(nb_wn) + 1/2)
    return wavenumbers
