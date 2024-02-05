"""
General purpose utilities
"""


from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common_utils.custom_vars import Opd, Wvn, Acq, Deg


@dataclass
class PlotOptions:
    figsize: tuple = (8, 6)
    fontsize: int = 25


def add_noise(
        array: np.ndarray[tuple[Opd | Wvn, Acq], np.dtype[np.float_]],
        snr_db: float,
) -> np.ndarray[tuple[Opd | Wvn, Acq], np.dtype[np.float_]]:
    snr_rms = 10 ** (snr_db / 20)
    noise_normal = np.random.randn(*array.shape)
    signal_std = array.std(axis=-2, keepdims=True)
    alpha = signal_std / snr_rms
    noise = alpha * noise_normal
    return array + noise


def rescale(
        array: np.ndarray[..., np.dtype[np.float_]],
        new_max: float = 1.,
        axis: int = -2,
) -> np.ndarray[..., np.dtype[np.float_]]:
    """Rescale an array to a new maximum, e.g., array_normed = array / max"""
    array_max = array.max(axis=axis, keepdims=True)
    array_normalized = array / array_max
    if new_max != 1.:
        array_normalized = array_normalized * new_max
    return array_normalized


def min_max_normalize(
        array: np.ndarray[..., np.dtype[np.float_]],
        new_min: float = 0.,
        new_max: float = 1.,
        axis: int = -2,
) -> np.ndarray[..., np.dtype[np.float_]]:
    """Normalize an array to be in the range [0, 1], e.g., array_normed = (array - min) / (max - min)"""
    array_min = array.min(axis=axis, keepdims=True)
    array_normalized = (array - array_min) / (array.max(axis=axis, keepdims=True) - array_min)
    if new_min != 0. or new_max != 1.:
        array_normalized = new_min + array_normalized * (new_max - new_min)
    return array_normalized


def standardize(
        array: np.ndarray[..., np.dtype[np.float_]],
        new_mean: float = 0.,
        new_std: float = 1.,
        axis: int = -2,
) -> np.ndarray[..., np.dtype[np.float_]]:
    """Standardize an array, e.g., array_std = (array - mean) / std."""
    array_mean = array.mean(axis=axis, keepdims=True)
    array_std = array.std(axis=axis, keepdims=True)
    array_standardized = (array - array_mean) / array_std
    if np.any(new_mean != 0) or np.any(new_std != 1.):
        array_standardized = new_std * array_standardized + new_mean
    return array_standardized


def match_stats(
        array: np.ndarray[..., np.dtype[np.float_]],
        reference: np.ndarray[..., np.dtype[np.float_]],
        axis: int = -2,
        is_rescale_reference: bool = False,
) -> tuple[np.ndarray[..., np.dtype[np.float_]], np.ndarray[..., np.dtype[np.float_]]]:
    """
    Used mostly for plot purposes, especially when comparing arrays.
    1- Rescale the reference if needed, to a maximum of 1
    2- Standardize the array to the statistics of the rescaled reference
    """
    if is_rescale_reference:
        reference = rescale(array=reference, axis=axis)
    ref_mean = reference.mean(axis=axis, keepdims=True)
    ref_std = reference.std(axis=axis, keepdims=True)
    array_standardized = standardize(array=array, new_mean=ref_mean, new_std=ref_std, axis=axis)
    return array_standardized, reference


def calculate_rmse(
        array: np.ndarray[tuple[..., Wvn, Acq], np.dtype[np.float_]],
        reference: np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]],
        is_match_stats: bool = False,
        is_rescale_reference: bool = False,
        is_match_axis: int = -2,
) -> np.ndarray[tuple[int, Acq], np.dtype[np.float_]]:
    """
    Calculate Normalized Root Mean Squared Error.
    """
    if is_match_stats:
        array, reference = match_stats(
            array=array,
            reference=reference,
            axis=is_match_axis,
            is_rescale_reference=is_rescale_reference,
        )
    error = array - reference
    error_vectorized = error.reshape(*error.shape[0:array.ndim-reference.ndim], -1)
    error_norm = np.linalg.norm(x=error_vectorized, ord=2, axis=-1)
    reference_norm = np.linalg.norm(x=reference)
    rmse = error_norm / reference_norm
    return rmse


def calculate_rmcw(
        monochromatic_array: np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]],
) -> float:
    """
    Ratio of Matching Central Wavenumbers:
      The maximum of each acquisition (column) is expected to match with the index of said acquisition.
    """
    acquisition_indices = np.arange(monochromatic_array.shape[-1])
    column_wise_argmax = np.argmax(monochromatic_array, axis=-2)
    rmcw = np.sum(column_wise_argmax == acquisition_indices) / monochromatic_array.shape[-1]
    return rmcw


def generate_shifted_dirac(array: np.ndarray, shift: float) -> np.ndarray:
    dirac_signal = np.zeros_like(array)
    index = index_from_value(array=array, value=shift)
    dirac_signal[index] = 1
    return dirac_signal


def index_from_value(array: np.ndarray, value: float) -> int:
    index = (np.abs(array - value)).argmin()
    return index


def generate_sampled_opds(nb_opd: int, opd_step: float, opd_min: float = 0) -> np.ndarray[tuple[Opd], np.dtype[np.float_]]:
    opds = opd_step * np.arange(nb_opd) + opd_min
    return opds


def generate_wavenumbers_from_opds(nb_wn: int, del_opd: float) -> np.ndarray[tuple[Wvn], np.dtype[np.float_]]:
    del_wn = 1 / (2 * nb_wn * del_opd)  # del_wn tends to zero as nb_wn tends to infinity (implies continuous)
    wavenumbers = del_wn * (np.arange(nb_wn) + 1/2)
    return wavenumbers


def convert_meter_units(values: float | np.ndarray, from_: str, to_: str):
    unit_to_meters = {
        "m": 1,
        "cm": 1e-2,
        "um": 1e-6,
        "nm": 1e-9,
    }

    if from_ not in unit_to_meters:
        raise ValueError(f"Unit {from_} is not supported")
    if to_ not in unit_to_meters:
        raise ValueError(f"Unit {to_} is not supported")

    value_in_meters = values * unit_to_meters[from_]
    converted_values = value_in_meters / unit_to_meters[to_]
    return converted_values


def polyval_rows(
        coefficients: np.ndarray[tuple[Opd, Deg], np.dtype[np.float_]],
        interval: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
) -> np.ndarray[tuple[Opd, Wvn], np.dtype[np.float_]]:
    powers = np.arange(coefficients.shape[1])[:, None]
    interval_powered = np.power(interval, powers)
    polynomials = coefficients @ interval_powered
    return polynomials
