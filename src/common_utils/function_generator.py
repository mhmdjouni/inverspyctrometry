from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
from scipy.fft import ifft, fft

from src.common_utils.custom_vars import Acq, Wvn


@dataclass
class FunctionGenerator(ABC):
    coefficients: np.ndarray[tuple[int, Acq], np.dtype[np.float_]]

    @abstractmethod
    def generate(
            self,
            variable: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]]:
        pass


@dataclass
class DiracGenerator(FunctionGenerator):
    shifts: np.ndarray[tuple[int, Acq], np.dtype[np.float_]]

    def generate(
            self,
            variable: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]]:
        distances_from_shifts = np.abs(self.shifts[:, :, None] - variable[None, None, :])
        shifts_idxs = np.argmin(distances_from_shifts, axis=-1)
        acq_idxs = np.arange(self.shifts.shape[-1])
        data = np.zeros(shape=(variable.size, self.shifts.shape[-1]))
        data[shifts_idxs[:, acq_idxs], acq_idxs] = self.coefficients[:, acq_idxs]
        return data


@dataclass
class CosineGenerator(FunctionGenerator):
    frequencies: np.ndarray[tuple[int, Acq], np.dtype[np.float_]]

    def generate(
            self,
            variable: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]]:
        variable_oscillated = 2 * np.pi * self.frequencies[:, :, None] * variable[None, None, :]
        cosine_funcs = np.cos(variable_oscillated)
        data = np.sum(self.coefficients[:, :, None] * cosine_funcs, axis=0).T
        return data


@dataclass
class GaussianGenerator(FunctionGenerator):
    means: np.ndarray[tuple[int, Acq], np.dtype[np.float_]]
    stds: np.ndarray[tuple[int, Acq], np.dtype[np.float_]]

    def rescale_parameters(
            self,
            ref_min,
            ref_max,
            new_min,
            new_max,
    ):
        means = new_min + (self.means - ref_min) / ref_max * (new_max - new_min)
        stds = self.stds / ref_max * (new_max - new_min)
        return replace(self, means=means, stds=stds)

    def generate(
            self,
            variable: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]]:
        variable_centered = (variable[None, None, :] - self.means[:, :, None]) / self.stds[:, :, None]
        gaussian_funcs = np.exp(- variable_centered ** 2)
        data = np.sum(self.coefficients[:, :, None] * gaussian_funcs, axis=0).T
        return data


@dataclass
class LorentzianGenerator(FunctionGenerator):
    locations: np.ndarray[tuple[int, Acq], np.dtype[np.float_]]  # location of the peak of the distribution
    scales: np.ndarray[tuple[int, Acq], np.dtype[np.float_]]  # half-width at half-maximum (HWHM)

    def generate(
            self,
            variable: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]]:
        variable_centered = (variable[None, None, :] - self.locations[:, :, None]) / self.scales[:, :, None]
        lorentzian_funcs = 1 / (1 + variable_centered ** 2)
        data = np.sum(self.coefficients[:, :, None] * lorentzian_funcs, axis=0).T
        return data
