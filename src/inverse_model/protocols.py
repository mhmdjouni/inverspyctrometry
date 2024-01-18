from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum
from src.common_utils.transmittance_response import TransmittanceResponse


@dataclass(frozen=True)
class InversionProtocol(ABC):

    @abstractmethod
    def reconstruct_spectrum(
            self,
            interferogram: Interferogram,
            transmittance_response: TransmittanceResponse,
    ) -> Spectrum:
        pass


@dataclass(frozen=True)
class PseudoInverse(InversionProtocol):

    def reconstruct_spectrum(
            self,
            interferogram: Interferogram,
            transmittance_response: TransmittanceResponse,
    ) -> Spectrum:
        compensation = 2 * (1 - 0.13)
        transmittance_response_compensated = transmittance_response.data / compensation - 1
        tr_pinv = np.linalg.pinv(transmittance_response_compensated)
        spectrum = tr_pinv @ interferogram.data
        return Spectrum(
            data=spectrum,
            wavenumbers=transmittance_response.wavenumbers,
        )
