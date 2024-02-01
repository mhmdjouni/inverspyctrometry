from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from src.common_utils.custom_vars import Opd, Acq
from src.common_utils.utils import add_noise, rescale, min_max_normalize, standardize


@dataclass(frozen=True)
class Interferogram:
    data: np.ndarray[tuple[Opd, Acq], np.dtype[np.float_]]
    opds: np.ndarray[tuple[Opd], np.dtype[np.float_]]
    opds_unit: str = "nm"

    def visualize(self, axs, acq_ind: int):
        axs.plot(self.opds, self.data[:, acq_ind])
        axs.set_title(rf"Interferogram")
        axs.set_ylabel("Intensity")
        axs.set_xlabel(rf"OPDs $\delta$ [{self.opds_unit}]")
        axs.grid()

    def add_noise(self, snr_db) -> Interferogram:
        noisy_data = add_noise(array=self.data, snr_db=snr_db)
        return replace(self, data=noisy_data)

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
