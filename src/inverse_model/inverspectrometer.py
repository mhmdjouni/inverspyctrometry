from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy import fft

from src.common_utils.custom_vars import Wvn
from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum


@dataclass(frozen=True)
class InverSpectrometer(ABC):
    transmittance: np.ndarray[tuple[Wvn], np.dtype[np.float_]]

    @abstractmethod
    def reconstruct_spectrum(
            self,
            interferogram: Interferogram,
    ) -> Spectrum:
        pass


class MichelsonInverSpectrometer(InverSpectrometer):

    def reconstruct_spectrum(
            self,
            interferogram: Interferogram,
    ) -> Spectrum:
        interferogram_compensated = interferogram.data - 1/2 * interferogram.data[0]
        spectrum = fft.idct(interferogram_compensated) / (2 * self.transmittance)
        # TODO: Fix the value of the field of wavenumbers
        return Spectrum(data=spectrum, wavenumbers=np.arange(interferogram.opds.size))
