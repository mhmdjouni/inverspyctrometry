from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum
from src.common_utils.transmittance_response import TransmittanceResponse
from src.common_utils.custom_vars import Wvn, Opd, InterferometerType
from src.common_utils.utils import calculate_phase_difference


def interferometer_factory(
        option: InterferometerType,
        transmittance: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
        opds: np.ndarray[tuple[Opd], np.dtype[np.float_]],
        reflectance: np.ndarray[tuple[Wvn], np.dtype[np.float_]] = None,
        order: int = 0,
) -> Interferometer:
    if option == InterferometerType.MICHELSON:
        interferometer = MichelsonInterferometer(transmittance=transmittance, opds=opds)
    elif option == InterferometerType.FABRY_PEROT:
        if reflectance is None:
            reflectance = 1 - transmittance
        interferometer = FabryPerotInterferometer(
            transmittance=transmittance,
            opds=opds,
            reflectance=reflectance,
            order=order,
        )
    else:
        raise ValueError(f"Option '{option.value}' is not supported")
    return interferometer


@dataclass(frozen=True)
class Interferometer(ABC):
    transmittance: np.ndarray[tuple[Wvn], np.dtype[np.float_]]
    opds: np.ndarray[tuple[Opd], np.dtype[np.float_]]

    @abstractmethod
    def transmittance_response(
            self,
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> TransmittanceResponse:
        pass

    def acquire_interferogram(
            self,
            spectrum: Spectrum,
    ) -> Interferogram:
        transmittance_response = self.transmittance_response(wavenumbers=spectrum.wavenumbers)
        return simulate_interferogram(transmittance_response=transmittance_response, spectrum=spectrum)

    def phase_difference(
            self,
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Opd, Wvn], np.dtype[np.float_]]:
        return calculate_phase_difference(opds=self.opds, wavenumbers=wavenumbers)


@dataclass(frozen=True)
class MichelsonInterferometer(Interferometer):

    def transmittance_response(
            self,
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> TransmittanceResponse:
        phase_difference = self.phase_difference(wavenumbers=wavenumbers)
        # This is equivalent to: y = (1/2 * y[0]) + 1/2 * scipy.fft.dct(2*T*x, type=2, norm=None)
        transmittance_response = 2 * self.transmittance[None, :] * (1 + np.cos(2 * np.pi * phase_difference))
        return TransmittanceResponse(
            data=transmittance_response,
            wavenumbers=wavenumbers,
            opds=self.opds,
        )


@dataclass(frozen=True)
class FabryPerotInterferometer(Interferometer):
    reflectance: np.ndarray[tuple[Wvn], np.dtype[np.float_]]
    order: int = 0

    def transmittance_response(
            self,
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> TransmittanceResponse:
        phase_difference = self.phase_difference(wavenumbers=wavenumbers)

        if self.order == 0:
            # Using the infinity-wave model and geometric series formula
            denominator = 1 + self.reflectance[None, :] ** 2 - 2 * self.reflectance[None, :] * np.cos(
                2 * np.pi * phase_difference)
            transmittance_response = self.transmittance[None, :] ** 2 * (1 / denominator)

        else:
            # Using the N-wave model approximation and the Poisson kernel formula
            q = 1 / (1 - self.reflectance ** 2)
            n_values = np.arange(1, self.order)
            reflectance_factors = self.reflectance[None, None, :] ** n_values[:, None, None]
            cosine_factors = np.cos(2 * np.pi * n_values[:, None, None] * phase_difference[None, :])
            series_sum = 1 + 2 * np.sum(reflectance_factors * cosine_factors, axis=0)
            transmittance_response = self.transmittance[None, :] ** 2 * (q[None, :] * series_sum)

        return TransmittanceResponse(
            data=transmittance_response,
            wavenumbers=wavenumbers,
            opds=self.opds,
        )


def simulate_interferogram(
        transmittance_response: TransmittanceResponse,
        spectrum: Spectrum,
) -> Interferogram:
    interferogram_data = transmittance_response.data @ spectrum.data
    return Interferogram(data=interferogram_data, opds=transmittance_response.opds)
