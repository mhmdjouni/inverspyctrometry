from dataclasses import dataclass

import numpy as np

from src.common_utils.custom_vars import Opd, Deg, InterferometerType, Wvn
from src.common_utils.transmittance_response import TransmittanceResponse
from src.common_utils.utils import polyval_rows
from src.direct_model.interferometer import Interferometer, interferometer_factory


@dataclass(frozen=True)
class Characterization:
    interferometer_type: InterferometerType
    transmittance_coefficients: np.ndarray[tuple[Opd, Deg], np.dtype[np.float_]]
    opds: np.ndarray[tuple[Opd], np.dtype[np.float_]]
    phase_shift: np.ndarray[tuple[Opd], np.dtype[np.float_]]
    reflectance_coefficients: np.ndarray[tuple[Opd, Deg], np.dtype[np.float_]]
    order: int

    def interferometer(self) -> Interferometer:
        interferometer = interferometer_factory(
            option=self.interferometer_type,
            transmittance_coefficients=self.transmittance_coefficients,
            opds=self.opds,
            phase_shift=self.phase_shift,
            reflectance_coefficients=self.reflectance_coefficients,
            order=self.order,
        )
        return interferometer

    def transmittance_response(
            self,
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> TransmittanceResponse:
        interferometer = self.interferometer()
        return interferometer.transmittance_response(wavenumbers=wavenumbers, is_correct_transmittance=True)

    def coeffs_to_polynomials(
            self,
            coefficients: np.ndarray[tuple[Opd, Deg], np.dtype[np.float_]],
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Opd, Wvn], np.dtype[np.float_]]:
        if coefficients.shape[0] == 1:
            coefficients = np.tile(coefficients.reshape(1, -1), (self.opds.size, 1))
            assert coefficients.ndim == 2
        return polyval_rows(coefficients=coefficients, interval=wavenumbers)

    def transmittance(
            self,
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Opd, Wvn], np.dtype[np.float_]]:
        return self.coeffs_to_polynomials(coefficients=self.transmittance_coefficients, wavenumbers=wavenumbers)

    def reflectance(
            self,
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Opd, Wvn], np.dtype[np.float_]]:
        return self.coeffs_to_polynomials(coefficients=self.reflectance_coefficients, wavenumbers=wavenumbers)
