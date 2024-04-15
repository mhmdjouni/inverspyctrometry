from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import numpy as np

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
class GaussianGenerator(FunctionGenerator):
    means: np.ndarray[tuple[int, Acq], np.dtype[np.float_]]
    stds: np.ndarray[tuple[int, Acq], np.dtype[np.float_]]

    def min_max_parameters(
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
        gaussian_funcs = np.exp(-variable_centered**2)
        data = np.sum(self.coefficients[:, :, None] * gaussian_funcs, axis=0).T
        return data


@dataclass
class CosineGenerator(FunctionGenerator):
    frequencies: np.ndarray[tuple[int, Acq], np.dtype[np.float_]]

    def generate(
            self,
            variable: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]]:
        pass


@dataclass
class LorentzianGenerator(FunctionGenerator):

    def generate(
            self,
            variable: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]]:
        pass


class DiracGenerator(FunctionGenerator):
    shifts: np.ndarray[tuple[int, Acq], np.dtype[np.float_]]

    def generate(
            self,
            variable: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]]:
        pass
