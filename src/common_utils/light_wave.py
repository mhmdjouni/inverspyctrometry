from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common_utils.custom_vars import Wvn, Acq


@dataclass(frozen=True)
class Spectrum:
    data: np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]]
    wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]]
    wavenumbers_unit: str = r"cm$^{-1}$"

    def visualize(self, axs, acq_ind: int):
        axs.plot(self.wavenumbers, self.data[:, acq_ind])
        axs.set_title(rf"Spectral Radiance")
        axs.set_ylabel("Intensity")
        axs.set_xlabel(rf"Wavenumbers $\sigma$ [{self.wavenumbers_unit}]")

    @classmethod
    def from_oscillations(
            cls,
            amplitudes: np.ndarray,
            opds: np.ndarray,
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> Spectrum:
        radiance = generate_radiance_from_oscillations(
            amplitudes=amplitudes,
            opds=opds,
            wavenumbers=wavenumbers,
        ).reshape((-1, 1))
        return Spectrum(data=radiance, wavenumbers=wavenumbers)


def generate_radiance_from_oscillations(
        amplitudes: np.ndarray,
        opds: np.ndarray,
        wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
) -> np.ndarray[tuple[Wvn], np.dtype[np.float_]]:
    if amplitudes.size != opds.size:
        raise ValueError("Amplitudes and OPDs must have the same size.")
    list_of_cosines = amplitudes[:, None] * np.cos(2 * np.pi * opds[:, None] * wavenumbers[None, :])
    radiance = np.sum(list_of_cosines, axis=0)
    return radiance
