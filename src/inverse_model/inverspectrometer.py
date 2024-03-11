from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy import fft

from src.common_utils.custom_vars import Wvn
from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum
from src.common_utils.utils import generate_wavenumbers_from_opds


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
        interferogram_compensated = interferogram.data - 1/2 * interferogram.data[0, :]
        # This is equivalent to: x = 2 * scipy.fft.idct(y - 1/2 * y[0], type=2. norm=None) / (2*T)
        spectrum = fft.idct(interferogram_compensated, axis=-2) / self.transmittance[:, None]
        wavenumbers = generate_wavenumbers_from_opds(
            wavenumbers_num=interferogram.opds.size,
            del_opd=np.mean(np.diff(interferogram.opds)),
        )
        return Spectrum(data=spectrum, wavenumbers=wavenumbers)
