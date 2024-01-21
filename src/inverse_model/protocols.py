from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from sklearn.decomposition import TruncatedSVD

from src.common_utils.custom_vars import InversionProtocolType
from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum
from src.common_utils.transmittance_response import TransmittanceResponse
from src.common_utils.utils import generate_wavenumbers_from_opds


def inversion_protocol_factory(option: InversionProtocolType, kwargs: dict):
    if option == InversionProtocolType.IDCT:
        return IDCT()

    elif option == InversionProtocolType.PSEUDO_INVERSE:
        return PseudoInverse()

    elif option == InversionProtocolType.TSVD:
        return TSVD(penalization_ratio=kwargs["penalization_ratio"])

    elif option == InversionProtocolType.RIDGE_REGRESSION:
        return RidgeRegression(penalization=kwargs["penalization"])

    elif option == InversionProtocolType.LORIS_VERHOEVEN:
        return LorisVerhoeven(penalization=kwargs["penalization"])

    elif option == InversionProtocolType.ADMM:
        return ADMM(penalization=kwargs["penalization"])

    else:
        raise ValueError(f"Inversion Protocol option {option} is not supported")


@dataclass(frozen=True)
class InversionProtocol(ABC):

    @abstractmethod
    def reconstruct_spectrum(
            self,
            interferogram: Interferogram,
            transmittance_response: TransmittanceResponse,
    ) -> Spectrum:
        pass


@dataclass(frozen=True)
class IDCT(InversionProtocol):
    """
    Inverse Discrete Cosine Transform
    """

    def reconstruct_spectrum(
            self,
            interferogram: Interferogram,
            transmittance_response: TransmittanceResponse,
    ) -> Spectrum:
        spectrum = fft.idct(interferogram.data)
        wavenumbers = generate_wavenumbers_from_opds(
            nb_wn=interferogram.opds.size,
            del_opd=np.mean(np.diff(interferogram.opds))
        )
        return Spectrum(data=spectrum, wavenumbers=wavenumbers)


@dataclass(frozen=True)
class PseudoInverse(InversionProtocol):

    def reconstruct_spectrum(
            self,
            interferogram: Interferogram,
            transmittance_response: TransmittanceResponse,
    ) -> Spectrum:
        tr_pinv = np.linalg.pinv(transmittance_response.data)
        spectrum = tr_pinv @ interferogram.data
        return Spectrum(
            data=spectrum,
            wavenumbers=transmittance_response.wavenumbers,
        )


@dataclass(frozen=True)
class TSVD(InversionProtocol):
    """
    Truncated Singular Value Decomposition
    """
    penalization_ratio: float

    def reconstruct_spectrum(
            self,
            interferogram: Interferogram,
            transmittance_response: TransmittanceResponse,
    ) -> Spectrum:
        u, s, v = np.linalg.svd(a=transmittance_response.data, full_matrices=False, compute_uv=True)
        n_components = int(s.size * self.penalization_ratio)
        s_penalized = 1 / s[:n_components]
        spectrum = (v[:n_components].T * s_penalized) @ u[:, :n_components].T @ interferogram.data
        return Spectrum(data=spectrum, wavenumbers=transmittance_response.wavenumbers)


@dataclass(frozen=True)
class RidgeRegression(InversionProtocol):
    penalization: float

    def reconstruct_spectrum(
            self,
            interferogram: Interferogram,
            transmittance_response: TransmittanceResponse,
    ) -> Spectrum:
        u, s, v = np.linalg.svd(a=transmittance_response.data, full_matrices=False, compute_uv=True)
        s_penalized = s / (s ** 2 + self.penalization ** 2)
        spectrum = (v.T * s_penalized) @ u.T @ interferogram.data
        return Spectrum(data=spectrum, wavenumbers=transmittance_response.wavenumbers)


@dataclass(frozen=True)
class LorisVerhoeven(InversionProtocol):
    penalization: float

    def reconstruct_spectrum(
            self,
            interferogram: Interferogram,
            transmittance_response: TransmittanceResponse,
    ) -> Spectrum:
        pass


@dataclass(frozen=True)
class ADMM(InversionProtocol):
    """
    Alternating Optimization Methods of Multipliers
    """
    penalization: float

    def reconstruct_spectrum(
            self,
            interferogram: Interferogram,
            transmittance_response: TransmittanceResponse,
    ) -> Spectrum:
        pass
