from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from src.common_utils.custom_vars import Opd, Acq
from src.common_utils.utils import add_noise


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

    def add_noise(self, snr_db):
        noisy_data = add_noise(array=self.data, snr_db=snr_db)
        return replace(self, data=noisy_data)
