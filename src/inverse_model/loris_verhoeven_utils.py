from dataclasses import dataclass, field

import numpy as np

from src.common_utils.custom_vars import Opd
from src.inverse_model.operators import LinearOperator, ProximalOperator


@dataclass(frozen=True)
class LorisVerhoevenIteration:
    transfer_matrix: LinearOperator
    domain_transform: LinearOperator
    prox_functional: ProximalOperator
    regularization_parameter: float
    observation: np.ndarray[tuple[Opd], np.dtype[np.float_]]

    tau: float = field(init=False)
    eta: float = field(init=False)
    rho: float = field(init=False)
    rho_tau: float = field(init=False)

    def __post_init__(self):
        # An alternative way for post-initializing dependent attributes of a frozen dataclass is to instantiate the
        #   object through an alternative constructor or from a factory function
        tau, eta = self.__convergence_params()
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "rho", 1.9)
        object.__setattr__(self, "rho_tau", self.rho * self.tau)

    def __convergence_params(self, tau: float = 1.0) -> tuple[float, float]:
        tau = tau / self.transfer_matrix.norm ** 2
        eta = 0.99 / tau / self.domain_transform.norm ** 2
        return tau, eta

    def update(
            self, prim: np.ndarray, dual: np.ndarray, error: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        error_prim = self.transfer_matrix.adjoint(error)
        prim_half = prim - self.tau * (error_prim + self.domain_transform.adjoint(dual))
        dual_half = self.prox_functional.proximal_conjugate(
            dual + self.eta * self.domain_transform.direct(prim_half),
            self.eta,
            self.regularization_parameter,
        )
        prim = prim - self.rho_tau * (error_prim + self.domain_transform.adjoint(dual_half))
        dual = dual + self.rho * (dual_half - dual)
        error = self.transfer_matrix.direct(prim) - self.observation
        return prim, dual, error
