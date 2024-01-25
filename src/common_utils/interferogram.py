from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common_utils.custom_vars import Opd
from src.common_utils.utils import add_noise


@dataclass(frozen=True)
class Interferogram:
    data: np.ndarray[tuple[Opd], np.dtype[np.float_]]
    opds: np.ndarray[tuple[Opd], np.dtype[np.float_]]
    opds_unit: str = "nm"

    def visualize(self, axs):
        axs.plot(self.opds, self.data)
        axs.set_title(rf"Interferogram")
        axs.set_ylabel("Intensity")
        axs.set_xlabel(rf"OPDs $\delta$ [{self.opds_unit}]")
