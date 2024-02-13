from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.common_utils.custom_vars import Wvn, Acq
from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum


def print_metrics(nb_tabs, category, best_idx, lambdaa, rmse_full, rmse_diagonal, rmcw, wavenumbers_size):
    tabs = "\t" * nb_tabs
    print(
        f"{tabs}"
        f"{category}: {best_idx:6},\t"
        f"Lambda: {lambdaa:7.4f},\t"
        f"RMSE: {rmse_full:.4f},\t"
        f"RMSE_DIAG: {rmse_diagonal:.4f},\t"
        f"RMCW: {rmcw:3}/{wavenumbers_size:3} ({rmcw / wavenumbers_size:6.4f})"
    )


def matching_central_wavenumbers_indicator(
        monochromatic_array: np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]],
) -> np.ndarray[bool]:
    acquisition_indices = np.arange(monochromatic_array.shape[-1])
    column_wise_argmax = np.argmax(monochromatic_array, axis=-2)
    is_matching_central_wavenumbers = np.equal(column_wise_argmax, acquisition_indices)
    return is_matching_central_wavenumbers  # Indicator function / vector


def calculate_rmcw(
        monochromatic_array: np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]],
) -> np.ndarray[..., np.dtype[np.int_]]:
    # TODO: Could be useful to have a MonochromaticSpectrum(Spectrum) class here
    """
    Number of Matching Central Wavenumbers:
      The maximum of each acquisition (column) is expected to match with the index of said acquisition.
    """
    acquisition_indices = np.arange(monochromatic_array.shape[-1])
    column_wise_argmax = np.argmax(monochromatic_array, axis=-2)
    rmcw = np.sum(column_wise_argmax == acquisition_indices, axis=-1, dtype=int)
    return rmcw


def split_matching_central_wavenumbers(
        monochromatic_array: np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]],
        target_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    # TODO: Could be useful to have a MonochromaticSpectrum(Spectrum) class here
    if target_type == "diagonal":
        target_array = np.diag(monochromatic_array)
    elif target_type == "maxima":
        target_array = np.max(monochromatic_array, axis=-2)
    else:
        raise ValueError(f"Split type '{target_type}' is not supported.")

    is_matching = matching_central_wavenumbers_indicator(monochromatic_array)
    matching = np.where(is_matching, target_array, np.nan)
    mismatching = np.where(~is_matching, target_array, np.nan)

    return matching, mismatching


def visualize_matching_central_wavenumbers(
        spectra: Spectrum,
        axs,
        target_type: str,
        linestyle: str = "-",
        title: str = None,
        xlabel: str = None,
        xlim: list = None,
        ylabel: str = None,
        ylim: list = None,
):
    # TODO: Could be useful to have a MonochromaticSpectrum(Spectrum) class here
    matching, mismatching = split_matching_central_wavenumbers(spectra.data, target_type=target_type)

    axs.plot(
        spectra.wavenumbers,
        np.ones(shape=spectra.data.shape[-1]),
        linestyle=linestyle,
        label="Reference",
        linewidth=3,
    )
    axs.plot(
        spectra.wavenumbers,
        matching,
        linestyle=linestyle,
        label="Matching",
    )
    axs.plot(
        spectra.wavenumbers,
        mismatching,
        linestyle=linestyle,
        label="Mismatching",
    )

    if title is None:
        title = "Matching Central Wavenumbers"
    if xlabel is None:
        xlabel = rf"Associated Central Wavenumbers [{spectra.wavenumbers_unit}]"
    if ylabel is None:
        ylabel = "Acquisition Intensity"
    if xlim is not None:
        axs.set_ylim(xlim)
    if ylim is not None:
        axs.set_ylim(ylim)

    axs.set_title(title)
    axs.set_ylabel(ylabel)
    axs.set_xlabel(xlabel)
    axs.legend()
    axs.grid(visible=True)


def crop_spectrum_wavenumbers(
        spectrum: Spectrum,
        new_range: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
) -> Spectrum:
    mask = (new_range[0] <= spectrum.wavenumbers) & (spectrum.wavenumbers <= new_range[-1])
    new_wavenumbers = spectrum.wavenumbers[mask]
    new_data = spectrum.data[mask][:, mask]
    return replace(spectrum, data=new_data, wavenumbers=new_wavenumbers)


def crop_interferogram_acquisitions(
        interferogram: Interferogram,
        central_wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
        new_range: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
) -> Interferogram:
    mask = (new_range[0] <= central_wavenumbers) & (central_wavenumbers <= new_range[-1])
    new_data = interferogram.data[:, mask]
    return replace(interferogram, data=new_data)
