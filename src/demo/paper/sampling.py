from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel
from tqdm import tqdm

from src.common_utils.custom_vars import InterferometerType, Wvn
from src.common_utils.transmittance_response import TransmittanceResponse
from src.direct_model.interferometer import interferometer_factory, Interferometer


class OPDSchema(BaseModel):
    num: int
    step: float

    def as_array(self) -> np.ndarray:
        return np.arange(0, self.num) * 0.2


class DeviceSchema(BaseModel):
    type: InterferometerType
    reflectance_scalar: float
    opds: OPDSchema

    def create(self) -> Interferometer:
        reflectance = np.array([self.reflectance_scalar])
        transmittance = 1. - reflectance
        return interferometer_factory(
            option=self.type,
            transmittance_coefficients=transmittance,
            opds=self.opds.as_array(),
            phase_shift=np.array([0.]),
            reflectance_coefficients=reflectance,
            order=0,
        )


class SpectralRangeSchema(BaseModel):
    min: float
    max: float


class SamplingOptionsSchema(BaseModel):
    device: DeviceSchema
    spectral_range: SpectralRangeSchema

    def create_experiment(self) -> SamplingExperiment:
        device = self.device.create()
        return SamplingExperiment(
            device_type=self.device.type,
            device=device,
            spectral_range=self.spectral_range,
        )


def dct_wn_sample(k: int, wn_step: float) -> float:
    """
    Calculate sigma_k = (k + 1/2) * sigma_step
      where k in [0, ..., K-1]
    """
    return (k + 1 / 2) * wn_step


def dct_wn_step(wn_num: int, opd_step: float) -> float:
    """
    Calculate sigma_step = 1 / (2 * K * delta_step)
    """
    return 1 / (2 * wn_num * opd_step)


def crop_interval(array: np.ndarray, min_lim: float = None, max_lim: float = None):
    if np.any(np.diff(array) < 0):
        raise ValueError("The array should be sorted in ascending order before cropping its limits.")

    if min_lim is not None:
        array = array[array >= min_lim]
    if max_lim is not None:
        array = array[array <= max_lim]
    return array


@dataclass
class SamplingExperiment:
    device_type: InterferometerType
    device: Interferometer
    spectral_range: SpectralRangeSchema

    def wavenumbers(self) -> np.ndarray[tuple[Wvn], np.dtype[np.float_]]:
        wn_num = int(self.device.opds.size * (self.device.harmonic_order() - 1))
        wn_step = dct_wn_step(wn_num, self.device.average_opd_step)
        wn_min = dct_wn_sample(k=0, wn_step=wn_step)
        wn_max = dct_wn_sample(k=wn_num - 1, wn_step=wn_step)
        wavenumbers = np.linspace(start=wn_min, stop=wn_max, num=wn_num, endpoint=True)
        wavenumbers = crop_interval(
            array=wavenumbers,
            min_lim=self.spectral_range.min,
            max_lim=self.spectral_range.max,
        )
        return wavenumbers

    def transfer_matrix(self):
        return self.device.transmittance_response(wavenumbers=self.wavenumbers())


def dct_orthogonalize(
        transfer_matrix: TransmittanceResponse,
        device_type: InterferometerType,
        reflectance: float,
) -> TransmittanceResponse:
    """This function compensates for the gains in each device then applies the operations that orthogonalize a DCT type-II matrix"""
    matrix = transfer_matrix.data
    if device_type == InterferometerType.MICHELSON:
        matrix = matrix - 2 * (1 - reflectance)
    elif device_type == InterferometerType.FABRY_PEROT:
        matrix = matrix / ((1 - reflectance) / (1 + reflectance)) - 1
    matrix /= np.sqrt(2 * matrix.shape[0])
    matrix[0] /= np.sqrt(2)
    return replace(transfer_matrix, data=matrix)


def visualize_all(
        fig,
        axs,
        transfer_matrix: TransmittanceResponse,
        opd_idx: Optional[int],
        linewidth: float,
        dct_orthogonalize_kwargs: Optional[dict],
):
    transfer_matrix.visualize(
        fig=fig,
        axs=axs[0, 0],
        title="",
        is_colorbar=True,
        x_ticks_num=5,
        x_ticks_decimals=1,
        y_ticks_decimals=0,
        aspect="auto",
    )

    transfer_matrix_ortho = dct_orthogonalize(
        transfer_matrix=transfer_matrix,
        **dct_orthogonalize_kwargs,
    )
    condition_number = transfer_matrix_ortho.condition_number()
    condition_number_str = f"{condition_number:.0f}" if condition_number < 1e10 else r"$\infty$"
    transfer_matrix_ortho.visualize_singular_values(
        axs=axs[0, 1],
        title=f"Condition number = {condition_number_str}",
        linewidth=linewidth,
        marker="o",
        markevery=5,
    )

    transfer_matrix.visualize_opd_response(
        axs=axs[1, 0],
        opd_idx=opd_idx,
        title=None,
        show_full_title=False,
        linewidth=linewidth,
    )

    transfer_matrix.visualize_dct(
        axs=axs[1, 1],
        opd_idx=opd_idx,
        title=None,
        show_full_title=False,
        linewidth=linewidth,
    )

    return fig, axs


def plot_harmonic_orders(
        fig,
        axs,
        threshold: float = 0.001,
        reflectivity_range: tuple = (0.0005, 0.85),
):
    reflectivity = np.arange(
        start=reflectivity_range[0],
        stop=reflectivity_range[1],
        step=0.0005,
    )
    order_float = np.log(threshold) / np.log(reflectivity) + 1
    order = np.ceil(order_float)
    axs.plot(reflectivity, order)
    axs.grid()
    axs.set_xlabel("Reflectivity")
    axs.set_ylabel("Harmonic Order")
    return fig, axs


def plot_condition_numbers(
        fig,
        axs,
        opd_schema: dict,
        reflectivity_range: tuple = (0.0005, 0.85, 0.01),
):
    reflectivities = np.arange(*reflectivity_range)

    condition_numbers = np.zeros_like(a=reflectivities)
    for i_rfl, reflectivity in tqdm(enumerate(reflectivities)):
        sampling_options_schema = {
            "device": {
                "type": InterferometerType.FABRY_PEROT,
                "reflectance_scalar": reflectivity,
                "opds": opd_schema,
            },
            "spectral_range": {
                "min": 1.,
                "max": 2.5,
            },
        }
        options = SamplingOptionsSchema(**sampling_options_schema)
        experiment = options.create_experiment()
        transfer_matrix_ortho = dct_orthogonalize(
            transfer_matrix=experiment.transfer_matrix(),
            device_type=options.device.type,
            reflectance=reflectivity,
        )
        condition_numbers[i_rfl] = transfer_matrix_ortho.condition_number()

    axs.plot(reflectivities, condition_numbers)
    axs.grid()
    axs.set_xlabel("Reflectivity")
    axs.set_ylabel("Condition number")
    return fig, axs
