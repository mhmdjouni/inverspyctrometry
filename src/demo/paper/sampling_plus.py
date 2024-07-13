from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from src.common_utils.custom_vars import InterferometerType, Opd, Wvn
from src.direct_model.interferometer import interferometer_factory, Interferometer


@dataclass(frozen=True)
class OPDSchema(BaseModel):
    num: int
    step: float

    def as_array(self) -> np.ndarray:
        return np.arange(0, self.num) * 0.2


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class SpectralRangeSchema(BaseModel):
    min: float
    max: float


@dataclass(frozen=True)
class SamplingOptionsSchema(BaseModel):
    device: DeviceSchema
    spectral_range: SpectralRangeSchema

    def create_experiment(self) -> SamplingExperiment:
        device = self.device.create()
        return SamplingExperiment(
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
    device: Interferometer
    spectral_range: SpectralRangeSchema

    @property
    def wavenumbers(self) -> np.ndarray[tuple[Wvn], np.dtype[np.float_]]:
        wn_num = self.device.opds.size * (self.device.harmonic_order() - 1)
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
        return self.device.transmittance_response(wavenumbers=self.wavenumbers)
