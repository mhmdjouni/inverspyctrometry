from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum
from src.common_utils.transmittance_response import TransmittanceResponse
from src.common_utils.custom_vars import Wvn, Opd, InterferometerType


@dataclass(frozen=True)
class Interferometer(ABC):
    transmittance: np.ndarray[tuple[Wvn], np.dtype[np.float_]]
    opds: np.ndarray[tuple[Opd], np.dtype[np.float_]]

    @abstractmethod
    def generate_transmittance_response(
            self,
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> TransmittanceResponse:
        pass

    def acquire_interferogram(
            self,
            spectrum: Spectrum,
    ) -> Interferogram:
        transmittance_response = self.generate_transmittance_response(wavenumbers=spectrum.wavenumbers)
        interferogram = transmittance_response.data @ spectrum.data
        return Interferogram(
            data=interferogram,
            opds=self.opds,
        )


@dataclass(frozen=True)
class MichelsonInterferometer(Interferometer):

    def generate_transmittance_response(
            self,
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> TransmittanceResponse:
        phase_difference = calculate_phase_difference(opds=self.opds, wavenumbers=wavenumbers)
        # This is equivalent to: y = (1/2 * y[0]) + 1/2 * scipy.fft.dct(2*T*x, type=2, norm=None)
        transmittance_response = 2 * self.transmittance[None, :] * (1 + np.cos(2 * np.pi * phase_difference))
        return TransmittanceResponse(
            data=transmittance_response,
            wavenumbers=wavenumbers,
            opds=self.opds,
        )


@dataclass(frozen=True)
class FabryPerotInterferometer(Interferometer):
    order: int = 0

    def generate_transmittance_response(
            self,
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> TransmittanceResponse:
        reflectance = 1 - self.transmittance
        phase_difference = self.opds[:, None] * wavenumbers[None, :]

        if self.order == 0:
            # Using the infinity-wave model and geometric series formula
            denominator = 1 + reflectance[None, :] ** 2 - 2 * reflectance[None, :] * np.cos(2 * np.pi * phase_difference)
            transmittance_response = self.transmittance[None, :] ** 2 * (1 / denominator)

        else:
            # Using the N-wave model approximation and the Poisson kernel formula
            q = 1 / (1 - reflectance ** 2)
            n_values = np.arange(1, self.order)
            reflectance_factors = reflectance[None, None, :] ** n_values[:, None, None]
            cosine_factors = np.cos(2 * np.pi * n_values[:, None, None] * phase_difference[None, :])
            series_sum = 1 + 2 * np.sum(reflectance_factors * cosine_factors, axis=0)
            transmittance_response = self.transmittance[None, :] ** 2 * (q[None, :] * series_sum)

        return TransmittanceResponse(
            data=transmittance_response,
            wavenumbers=wavenumbers,
            opds=self.opds,
        )


def calculate_phase_difference(
        opds: np.ndarray[tuple[Opd], np.dtype[np.float_]],
        wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
) -> np.ndarray[tuple[Opd, Wvn], np.dtype[np.float_]]:
    return opds[:, None] * wavenumbers[None, :]


def interferometer_factory(
        option: InterferometerType,
        transmittance: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
        opds: np.ndarray[tuple[Opd], np.dtype[np.float_]],
        order: int = 0,
) -> Interferometer:
    # TODO: Parameters to be optimized?
    if option == InterferometerType.MICHELSON:
        interferometer = MichelsonInterferometer(transmittance=transmittance, opds=opds)
    elif option == InterferometerType.FABRY_PEROT:
        interferometer = FabryPerotInterferometer(transmittance=transmittance, opds=opds, order=order)
    else:
        raise ValueError(f"Option '{option.value}' is not supported")
    return interferometer
