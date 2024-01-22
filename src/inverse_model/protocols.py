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
from src.inverse_model.operators import ProximalOperator, LinearOperator
from src.inverse_model.loris_verhoeven_utils import LorisVerhoevenIteration


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
        lsv, sv, rsv = np.linalg.svd(a=transmittance_response.data, full_matrices=False, compute_uv=True)
        nb_sv = int(sv.size * self.penalization_ratio)
        sv_penalized = 1 / sv[:nb_sv]
        spectrum = (rsv[:nb_sv].T * sv_penalized) @ lsv[:, :nb_sv].T @ interferogram.data
        return Spectrum(data=spectrum, wavenumbers=transmittance_response.wavenumbers)


@dataclass(frozen=True)
class RidgeRegression(InversionProtocol):
    penalization: float

    def reconstruct_spectrum(
            self,
            interferogram: Interferogram,
            transmittance_response: TransmittanceResponse,
    ) -> Spectrum:
        lsv, sv, rsv = np.linalg.svd(a=transmittance_response.data, full_matrices=False, compute_uv=True)
        sv_penalized = sv / (sv ** 2 + self.penalization ** 2)
        spectrum = (rsv.T * sv_penalized) @ lsv.T @ interferogram.data
        return Spectrum(data=spectrum, wavenumbers=transmittance_response.wavenumbers)


@dataclass(frozen=True)
class LorisVerhoeven(InversionProtocol):
    regularization_parameter: float
    prox_functional: ProximalOperator
    domain_transform: LinearOperator
    nb_iters: int

    def reconstruct_spectrum(
            self,
            interferogram: Interferogram,
            transmittance_response: TransmittanceResponse,
    ) -> Spectrum:
        transfer_matrix = LinearOperator.from_matrix(matrix=transmittance_response.data)

        lv_iter = LorisVerhoevenIteration(
            transfer_matrix=transfer_matrix,
            domain_transform=self.domain_transform,
            prox_functional=self.prox_functional,
            regularization_parameter=self.regularization_parameter,
            observation=interferogram.data,
        )
        prim = transfer_matrix.adjoint(interferogram.data)
        dual = self.domain_transform.direct(prim)
        error = transfer_matrix.direct(prim) - interferogram.data
        for q in range(self.nb_iters):
            prim, dual, error = lv_iter.update(prim, dual, error)

        return prim


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
