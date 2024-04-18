from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy import fft
import matplotlib.pyplot as plt

from src.common_utils.custom_vars import Wvn, Opd, Acq
from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum
from src.common_utils.utils import generate_wavenumbers_from_opds
from src.outputs.visualization import imshow_custom


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

    def reconstruct_spectrum(
            self,
            interferogram: Interferogram,
    ) -> Spectrum:
        if self.is_mean_center:
            interferogram = interferogram.center(new_mean=0, axis=-2)
        interferogram_dft = np.abs(fft.fft(interferogram.data, axis=-2, norm="forward")[0:interferogram.data.shape[-2] // 2, :])
        interferogram_dft = (interferogram_dft - interferogram_dft[1]) / 0.187
        # interferogram_dft = fft.idct(interferogram.data, axis=-2, norm="backward") / (0.1 * np.sqrt(3))
        wn_step_dct = 1 / (2 * interferogram.opds.max()) * 2
        wavenumbers_dct = generate_wavenumbers_from_opds(
            wavenumbers_num=interferogram_dft.shape[-2],
            del_opd=np.mean(np.diff(interferogram.opds)),
        )
        wns_range_condition = np.logical_and(
            self.wavenumbers[0] <= wavenumbers_dct,
            wavenumbers_dct <= self.wavenumbers[-1]
        )
        wn_idx_range = np.where(wns_range_condition)
        kernel_coefficients = self.kernel_fourier_coefficients()

        transfer_matrix = self.equate_coefficients(
            interferogram_dft_size=interferogram_dft.shape[-2],
            wn_idx_start=wn_idx_range[0][0],
            wn_idx_stop=wn_idx_range[0][-1],
            order=self.order,
            kernel_fourier_coefficients=kernel_coefficients,
        )

        # fig, axs = plt.subplots(1, 1, squeeze=False)
        # imshow_custom(
        #     fig=fig,
        #     axs=axs[0, 0],
        #     image=transfer_matrix,
        #     title=f"Harmonics Order M = {self.order}",
        #     x_variable=wavenumbers_dct,
        #     y_variable=wavenumbers_dct,
        #     x_label="Wavenumbers [cm^-1]",
        #     y_label="Wavenumbers [cm^-1]",
        #     vmin=0,
        #     vmax=0.01,
        #     interpolation="nearest",
        # )

        spectrum_coefficients = self.recursive_recovery(
            interferogram_dft_amplitudes=interferogram_dft,
            transfer_matrix=transfer_matrix,
            wn_idx_start=wn_idx_range[0][0],
            wn_idx_stop=wn_idx_range[0][-1],
            kernel_coefficient_1=kernel_coefficients[1],
        )

        wavenumbers_target = wavenumbers_dct[wn_idx_range]
        spectrum = spectrum_from_haar(
            wavenumbers=wavenumbers_target,
            wn_idx_range=wn_idx_range[0],
            spectrum_coefficients=spectrum_coefficients,
            wn_step=wn_step_dct,
        )

        print(f"M = {self.order}")
        print(f"A = {np.round(kernel_coefficients, 3)}")
        print(f"B =\n{np.round(transfer_matrix[:7, :7], 3)}")
        print("\n\n")

        return spectrum

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
        for dft_idx in wn_idx_range:
            for wn_idx in wn_idx_range:
                for order_idx in order_idx_range:
                    sub_order_increment_range = list(range(order_idx))
                    for sub_order_increment in sub_order_increment_range:
                        if order_idx * wn_idx + sub_order_increment == dft_idx:
                            coefficient_ratio = kernel_fourier_coefficients[order_idx] / order_idx
                            transfer_matrix[dft_idx, wn_idx] += coefficient_ratio

        return transfer_matrix

    @staticmethod
    def recursive_recovery(
            interferogram_dft_amplitudes: np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]],  # J
            transfer_matrix: np.ndarray[tuple[Wvn, Wvn], np.dtype[np.float_]],  # B
            wn_idx_start: int,  # k1
            wn_idx_stop: int,  # k2
            kernel_coefficient_1: float,  # A1
    ):
        wn_idx_range = np.arange(start=wn_idx_start, stop=wn_idx_stop + 1)
        spectrum_amplitudes = np.zeros_like(interferogram_dft_amplitudes)

        tau = wn_idx_range[0]
        spectrum_amplitudes[tau] = interferogram_dft_amplitudes[tau]

        for tau in wn_idx_range[1:]:
            spectrum_amplitudes_cropped = spectrum_amplitudes[wn_idx_start:tau, :]
            transfer_matrix_row_cropped = transfer_matrix[tau, wn_idx_start:tau][:, None]
            gamma = np.sum(spectrum_amplitudes_cropped * transfer_matrix_row_cropped, axis=-2, keepdims=True)
            spectrum_amplitudes[tau] = (interferogram_dft_amplitudes[tau] - gamma) / (2 * kernel_coefficient_1)

        return spectrum_amplitudes


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
