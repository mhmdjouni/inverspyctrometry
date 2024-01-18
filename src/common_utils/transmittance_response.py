from dataclasses import dataclass

import numpy as np
from scipy import fft

from src.common_utils.custom_vars import Opd, Wvn


@dataclass(frozen=True)
class TransmittanceResponse:
    data: np.ndarray[tuple[Opd, Wvn], np.dtype[np.float_]] | np.ndarray
    wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]]
    opds: np.ndarray[tuple[Opd], np.dtype[np.float_]] | np.ndarray
    wavenumbers_unit: str = r"cm$^{-1}$"
    opds_unit: str = "nm"

    def compute_singular_values(self):
        zero_mean_opd_responses = self.data - np.mean(self.data, axis=1, keepdims=True)
        sing_vals = np.linalg.svd(a=zero_mean_opd_responses, full_matrices=False, compute_uv=False)
        return sing_vals

    def compute_dct(self, opd_idx: int = -1):
        zero_mean_opd_responses = self.data - np.mean(self.data, axis=1, keepdims=True)
        if opd_idx == -1:
            array = np.sum(zero_mean_opd_responses, axis=0)
        else:
            array = zero_mean_opd_responses[opd_idx]
        dct = fft.dct(x=array, type=2, norm='ortho')
        return dct

    def visualize(self, axs):
        axs.imshow(self.data, aspect='auto')

        opd_ticks = np.linspace(start=0, stop=self.opds.size-1, num=10, dtype=int)
        opd_labels = np.around(a=self.opds[opd_ticks], decimals=2)
        axs.set_yticks(ticks=opd_ticks, labels=opd_labels)

        wavenumbers_ticks = np.linspace(start=0, stop=self.wavenumbers.size-1, num=10, dtype=int)
        wavenumbers_labels = np.around(a=self.wavenumbers[wavenumbers_ticks], decimals=2)
        axs.set_xticks(ticks=wavenumbers_ticks, labels=wavenumbers_labels)

        axs.set_title("Transmittance Response")
        axs.set_ylabel(rf"OPDs $\delta$ [{self.opds_unit}]")
        axs.set_xlabel(rf"Wavenumbers $\sigma$ [{self.wavenumbers_unit}]")

    def visualize_opd_response(self, axs, opd_idx):
        axs.plot(self.wavenumbers, self.data[opd_idx])
        axs.set_title(rf"OPD Response at $\delta$={self.opds[opd_idx]} {self.opds_unit}")
        axs.set_ylabel("Intensity")
        axs.set_xlabel(rf"Wavenumbers $\sigma$ [{self.wavenumbers_unit}]")

    def visualize_wavenumber_response(self, axs, wavenumber_idx):
        axs.plot(self.opds, self.data[:, wavenumber_idx])
        axs.set_title(rf"Wavenumber Response at $\sigma$={self.wavenumbers[wavenumber_idx]} {self.wavenumbers_unit}")
        axs.set_ylabel("Intensity")
        axs.set_xlabel(rf"OPDs $\delta$ [{self.opds_unit}]")

    def visualize_singular_values(self, axs):
        singular_values = self.compute_singular_values()
        axs.plot(singular_values)
        axs.set_title("Singular Values")
        axs.set_ylabel("Amplitude")
        axs.set_xlabel(r"Singular Value index [$R_{A}$]")

    def visualize_dct(self, axs, opd_idx: int = -1):
        dct = self.compute_dct(opd_idx)
        x_axis = np.mean(np.diff(a=self.opds)) * np.arange(dct.shape[0])
        axs.plot(x_axis, dct)
        if opd_idx == -1:
            opd_idx_str = 'all oscillations'
        else:
            opd_idx_str = rf'OPD Response at $\delta$={self.opds[opd_idx]} {self.opds_unit}'
        axs.set_title(f"DCT of {opd_idx_str}")
        axs.set_ylabel("Amplitude")
        axs.set_xlabel(rf"Fourier Oscillations $\delta$ [{self.opds_unit}]")
