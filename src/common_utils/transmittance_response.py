from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from matplotlib.figure import Figure
from scipy import fft

from src.common_utils.custom_vars import Opd, Wvn
from src.common_utils.utils import rescale


@dataclass(frozen=True)
class TransmittanceResponse:
    data: np.ndarray[tuple[Opd, Wvn], np.dtype[np.float_]] | np.ndarray
    wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]]
    opds: np.ndarray[tuple[Opd], np.dtype[np.float_]] | np.ndarray
    wavenumbers_unit: str = r"cm$^{-1}$"
    opds_unit: str = "nm"

    def rescale(self, new_max: float = 1., axis: int = None) -> TransmittanceResponse:
        rescaled_data = rescale(array=self.data, new_max=new_max, axis=axis)
        return replace(self, data=rescaled_data)

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

    def visualize(
            self,
            fig: Figure,
            axs,
            title: str = None,
            aspect: str | float = 1.,
            vmin: float = None,
            vmax: float = None,
            is_colorbar: bool = True,
            x_ticks_num: int = 6,
            x_ticks_decimals: int = 2,
            y_ticks_num: int = 6,
            y_ticks_decimals: int = 2,
    ):
        imshow = axs.imshow(
            self.data,
            vmin=vmin,
            vmax=vmax,
        )
        if is_colorbar:
            fig.colorbar(imshow, ax=axs)

        opd_ticks = np.linspace(start=0, stop=self.opds.size - 1, num=y_ticks_num, dtype=int)
        opd_labels = np.around(a=self.opds[opd_ticks], decimals=y_ticks_decimals)
        axs.set_yticks(ticks=opd_ticks, labels=opd_labels)

        wavenumbers_ticks = np.linspace(start=0, stop=self.wavenumbers.size-1, num=x_ticks_num, dtype=int)
        wavenumbers_labels = np.around(a=self.wavenumbers[wavenumbers_ticks], decimals=x_ticks_decimals)
        axs.set_xticks(ticks=wavenumbers_ticks, labels=wavenumbers_labels)

        if title is None:
            title = "Transmittance Response"
        axs.set_title(title)
        axs.set_aspect(aspect)
        axs.set_ylabel(rf"OPDs $\delta$ [{self.opds_unit}]")
        axs.set_xlabel(rf"Wavenumbers $\sigma$ [{self.wavenumbers_unit}]")

    def visualize_opd_response(
            self,
            axs,
            opd_idx: int,
            title: str = None,
            show_full_title: bool = True,
            linewidth: float = 1.5,
    ):
        """Visualize a selected row of the transmittance response"""
        axs.plot(
            self.wavenumbers,
            self.data[opd_idx],
            linewidth=linewidth,
        )
        if title is None:
            if show_full_title:
                title = rf"OPD Response for $\delta$={self.opds[opd_idx]:.2f} {self.opds_unit}"
            else:
                title = rf"$\delta$={self.opds[opd_idx]:.2f} {self.opds_unit}"
        axs.set_title(title)
        axs.set_ylabel("Intensity")
        axs.set_xlabel(rf"Wavenumbers $\sigma$ [{self.wavenumbers_unit}]")
        axs.grid(True)

    def visualize_wavenumber_response(
            self,
            axs,
            wavenumber_idx,
            linewidth: float = 1.5,
    ):
        axs.plot(self.opds, self.data[:, wavenumber_idx], linewidth=linewidth)
        axs.set_title(rf"Wavenumber Response for $\sigma$={self.wavenumbers[wavenumber_idx]} {self.wavenumbers_unit}")
        axs.set_ylabel("Intensity")
        axs.set_xlabel(rf"OPDs $\delta$ [{self.opds_unit}]")
        axs.grid(True)

    def visualize_singular_values(
            self,
            axs,
            title: str = None,
            linewidth: float = 1.5,
    ):
        singular_values = self.compute_singular_values()
        axs.plot(singular_values, linewidth=linewidth)
        if title is None:
            title = "Singular Values"
        axs.set_title(title)
        axs.set_ylabel("Amplitude")
        axs.set_xlabel(r"Singular Value index")
        axs.grid(True)

    def visualize_dct(
            self,
            axs,
            opd_idx: int = -1,
            title: str = None,
            show_full_title: bool = True,
            linewidth: float = 1.5,
    ):
        dct = self.compute_dct(opd_idx)
        x_axis = np.mean(np.diff(a=self.opds)) * np.arange(dct.shape[0])
        axs.plot(x_axis, dct, linewidth=linewidth)
        if title is None:
            if show_full_title:
                if opd_idx == -1:
                    opd_idx_str = 'all oscillations'
                else:
                    opd_idx_str = rf'OPD Response for $\delta$={self.opds[opd_idx]:.2f} {self.opds_unit}'
                title = f"DCT of {opd_idx_str}"
            else:
                if opd_idx == -1:
                    opd_idx_str = 'All oscillations'
                else:
                    opd_idx_str = rf'$\delta$={self.opds[opd_idx]:.2f} {self.opds_unit}'
                title = f"{opd_idx_str}"
        axs.set_title(title)
        axs.set_ylabel("Amplitude")
        axs.set_xlabel(rf"Fourier Oscillations $\delta$ [{self.opds_unit}]")
        axs.grid(True)
