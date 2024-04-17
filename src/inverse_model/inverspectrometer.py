from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy import fft

from src.common_utils.custom_vars import Wvn, Opd, Acq
from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum
from src.common_utils.utils import generate_wavenumbers_from_opds


# TODO: Supports only constant transmittance and reflectance
#  (otherwise, we would need information on the coefficients and the wavenumbers)


@dataclass(frozen=True)
class InverSpectrometer(ABC):
    transmittance: np.ndarray[tuple[Wvn], np.dtype[np.float_]]
    wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]]

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


@dataclass(frozen=True)
class FabryPerotInverSpectrometerHaar(InverSpectrometer):
    """
    This method is based on expanding the DFT of the interferogram and the spectrum by a Haar or box function.
    """
    reflectance: np.ndarray[tuple[Wvn], np.dtype[np.float_]]
    order: int
    is_mean_center: bool = True

    def variable(
            self,
            opds: np.ndarray[tuple[Opd], np.dtype[np.float_]],
    ) -> np.ndarray:
        return opds[:, None] * self.wavenumbers[None, :]

    def kernel(
            self,
            variable: np.ndarray,
    ) -> np.ndarray:
        numerator = self.transmittance ** 2
        phase_difference = 2 * np.pi * variable
        denominator = 1 + self.reflectance ** 2 - 2 * self.reflectance * np.cos(phase_difference)
        return numerator / denominator

    def kernel_fourier_coefficients(self) -> np.ndarray:
        # TODO: For some reason, to match the results in the paper,
        #  the coefficients for n>0 are not multiplied by 2
        variable = np.linspace(start=0, stop=1, num=int(1e4), endpoint=False)

        kernel = self.kernel(variable=variable)
        n_values = np.arange(self.order + 1)
        cosines = np.cos(2 * np.pi * n_values[:, None] * variable[None, :])

        integrands = kernel[None, :] * cosines  # This should be multiplied by 2 in principle for n>0
        variable_differential = np.mean(np.diff(variable))  # a scalar, because it's regularly sampled
        coefficients = 2 * np.sum(integrands * variable_differential, axis=-1)
        coefficients[0] /= 2

        return coefficients

    def reconstruct_spectrum(
            self,
            interferogram: Interferogram,
    ) -> Spectrum:
        if self.is_mean_center:
            interferogram = interferogram.center(new_mean=0, axis=-2)

        interferogram_dct = fft.dct(interferogram.data, axis=-2)
        wn_step_dct = 1 / (2 * interferogram.opds.max())
        wavenumbers_dct = np.arange(interferogram.opds.size) * wn_step_dct
        wns_range_condition = np.logical_and(
            self.wavenumbers[0] <= wavenumbers_dct,
            wavenumbers_dct <= self.wavenumbers[-1]
        )
        wn_idx_range = np.where(wns_range_condition)
        kernel_coefficients = self.kernel_fourier_coefficients()

        transfer_matrix = self.equate_coefficients(
            interferogram_dft_size=interferogram_dct.shape[-2],
            wn_idx_start=wn_idx_range[0][0],
            wn_idx_stop=wn_idx_range[0][-1],
            order=self.order,
            kernel_fourier_coefficients=kernel_coefficients,
        )

        spectrum_coefficients = self.recursive_recovery(
            interferogram_dct=interferogram_dct,
            transfer_matrix=transfer_matrix,
            wn_idx_start=wn_idx_range[0][0],
            wn_idx_stop=wn_idx_range[0][-1],
            kernel_coefficient=kernel_coefficients[1],
        )

        wavenumbers_target = wavenumbers_dct[wn_idx_range]
        spectrum = spectrum_from_haar(
            wavenumbers=wavenumbers_target,
            wn_idx_range=wn_idx_range[0],
            spectrum_coefficients=spectrum_coefficients,
            wn_step=wn_step_dct,
        )

        return spectrum

    # TODO: Move these to Haar inversion utils?
    @staticmethod
    def equate_coefficients(
            interferogram_dft_size: int,
            wn_idx_start: int,
            wn_idx_stop: int,
            order: int,
            kernel_fourier_coefficients: np.ndarray,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        """
        TODO: The result doesn't match with the paper
        """
        transfer_matrix = np.zeros((interferogram_dft_size, interferogram_dft_size))
        wn_idx_range = list(range(wn_idx_start, wn_idx_stop + 1))
        order_idx_range = list(range(1, order + 1))
        for dft_idx in wn_idx_range:  # u in [k_1, k_2]
            for wn_idx in wn_idx_range:  # k in [k_1, k_2]
                for order_idx in order_idx_range:  # n in [1, M]
                    for sub_order_increment in range(order_idx):  # i in [0, n-1]
                        index_factor = order_idx * wn_idx + sub_order_increment
                        if index_factor == dft_idx:
                            coefficient_ratio = kernel_fourier_coefficients[order_idx] / order_idx
                            transfer_matrix[dft_idx, wn_idx] += coefficient_ratio
        return transfer_matrix

    @staticmethod
    def recursive_recovery(
            interferogram_dct: np.ndarray,
            transfer_matrix: np.ndarray,
            wn_idx_start: int,
            wn_idx_stop: int,
            kernel_coefficient: float,
    ):
        wn_idx_range = list(range(wn_idx_start, wn_idx_stop + 1))
        spectrum_coefficients = np.zeros_like(interferogram_dct)
        dft_idx = wn_idx_range[0]
        spectrum_coefficients[dft_idx] = interferogram_dct[dft_idx]
        for dft_idx in wn_idx_range[1:]:
            vect = spectrum_coefficients[wn_idx_range[0]:dft_idx] * transfer_matrix[wn_idx_range[0]:dft_idx, [dft_idx]]
            gamma = np.sum(vect, axis=-2, keepdims=True)
            spectrum_coefficients[dft_idx] = (interferogram_dct[dft_idx] - gamma) / (2 * kernel_coefficient)
        return spectrum_coefficients


def evaluate_haar_function(
        array: np.ndarray,
        shift: float = 0,
        dilation: float = 1,
) -> np.ndarray:
    """
    Phi((t-s)/d)  =>  [0 <= (t-s)/d < 1]  =>  [s <= t < s + d]
    """
    phi = np.where(np.logical_and(shift <= array, array < shift + dilation), 1, 0)
    return phi


def spectrum_from_haar(
        wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
        spectrum_coefficients: np.ndarray[tuple[int, Acq], np.dtype[np.float_]],
        wn_idx_range: np.ndarray[tuple[int], np.dtype[np.int_]],
        wn_step: float,
) -> Spectrum:
    shifts = wn_idx_range * wn_step
    haar_functions = evaluate_haar_function(
        array=wavenumbers[None, None, :],
        shift=shifts[:, None, None],
        dilation=wn_step
    )
    spectrum_data = np.sum(spectrum_coefficients[wn_idx_range, :, None] * haar_functions, axis=0).T
    return Spectrum(data=spectrum_data, wavenumbers=wavenumbers)
