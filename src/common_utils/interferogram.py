from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from matplotlib.figure import Figure

from src.common_utils.custom_vars import Opd, Acq
from src.common_utils.utils import add_noise, rescale, min_max_normalize, standardize, center, match_stats


@dataclass(frozen=True)
class Interferogram:
    data: np.ndarray[tuple[Opd, Acq], np.dtype[np.float_]]
    opds: np.ndarray[tuple[Opd], np.dtype[np.float_]]
    opds_unit: str = "nm"

    def visualize(
            self,
            axs,
            acq_ind: int,
            is_sort_opds: bool = True,
            linestyle: str = "-",
            label: str = None,
            color: str = "C0",
            linewidth: float = 1.5,
            title: str = None,
            ylabel: str = None,
            ylim: list = None,
    ):
        if is_sort_opds:
            new_indices = np.argsort(self.opds)
        else:
            new_indices = np.arange(self.opds.size, dtype=int)
        axs.plot(
            self.opds[new_indices],
            self.data[new_indices, acq_ind],
            linestyle=linestyle,
            label=label,
            color=color,
            linewidth=linewidth,
        )
        if title is None:
            title = "Interferogram"
        if ylabel is None:
            ylabel = "Intensity"
        if ylim is not None:
            axs.set_ylim(ylim)

        axs.set_title(title)
        axs.set_ylabel(ylabel)
        axs.set_xlabel(rf"OPDs $\delta$ [{self.opds_unit}]")
        axs.legend()
        axs.grid(visible=True)

    def visualize_matrix(
            self,
            fig: Figure,
            axs,
            title: str = None,
            vmin: float = None,
            vmax: float = None
    ):
        pos = axs.imshow(
            self.data,
            # aspect='auto',
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(pos, ax=axs)

        opd_ticks = np.linspace(start=0, stop=self.opds.size - 1, num=6, dtype=int)
        opd_labels = np.around(a=self.opds[opd_ticks], decimals=2)
        axs.set_yticks(ticks=opd_ticks, labels=opd_labels)

        if title is None:
            title = "Interferogram Acquisitions"
        axs.set_title(title)
        axs.set_ylabel(rf"OPDs $\delta$ [{self.opds_unit}]")
        axs.set_xlabel(r"Acquisitions index $n \in \{1, \dots, N\}$")

    def add_noise(self, snr_db) -> Interferogram:
        noisy_data = add_noise(array=self.data, snr_db=snr_db)
        return replace(self, data=noisy_data)

    def center(self, new_mean: float = 0., axis: int = -2) -> Interferogram:
        centered_data = center(array=self.data, new_mean=new_mean, axis=axis)
        return replace(self, data=centered_data)

    def rescale(self, new_max: float = 1., axis: int = -2) -> Interferogram:
        rescaled_data = rescale(array=self.data, new_max=new_max, axis=axis)
        return replace(self, data=rescaled_data)

    def min_max_normalize(
            self,
            new_min: float = 0.,
            new_max: float = 1.,
            axis: int = -2
    ) -> Interferogram:
        normalized_data = min_max_normalize(array=self.data, new_min=new_min, new_max=new_max, axis=axis)
        return replace(self, data=normalized_data)

    def standardize(
            self,
            new_mean: float = 0.,
            new_std: float = 1.,
            axis: int = -2
    ) -> Interferogram:
        standardized_data = standardize(array=self.data, new_mean=new_mean, new_std=new_std, axis=axis)
        return replace(self, data=standardized_data)

    def match_stats(
            self,
            reference: Interferogram,
            axis: int = -2,
            is_rescale_reference: bool = False,
    ) -> tuple[Interferogram, Interferogram]:
        matched_data, scaled_reference = match_stats(
            array=self.data,
            reference=reference.data,
            axis=axis,
            is_rescale_reference=is_rescale_reference,
        )
        return replace(self, data=matched_data), replace(reference, data=scaled_reference)
