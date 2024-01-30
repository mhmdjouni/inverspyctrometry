from typing import Optional, Sequence, List

import numpy as np
from pydantic import BaseModel, RootModel

from src.common_utils.custom_vars import InversionProtocolType, NormOperatorType, LinearOperatorMethod, \
    RegularizationParameterKey, RegularizationParameterListSpace
from src.inverse_model.operators import CTVOperator, NormOperator, LinearOperator


class LambdaasSchema(BaseModel):
    key: RegularizationParameterKey
    start: float
    stop: float
    num: int
    space: RegularizationParameterListSpace

    def as_array(self) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
        if self.space == RegularizationParameterListSpace.LINSPACE:
            lambdaas = np.linspace(start=self.start, stop=self.stop, num=self.num)
        elif self.space == RegularizationParameterListSpace.LOGSPACE:
            lambdaas = np.logspace(start=self.start, stop=self.stop, num=self.num)
        elif self.space == RegularizationParameterListSpace.NOT_APPLICABLE:
            lambdaas = np.zeros(shape=self.num)
        else:
            raise ValueError(f"Option {self.space} is not supported. Only 'linspace' and 'logspace' are supported.")
        return lambdaas


class InversionProtocolExperimentSchema(BaseModel):
    """
    Schema for inversion protocol inputs used to launch experiments, not to directly initialize an Inversion Protocol
    """
    id: int
    title: str
    type: InversionProtocolType
    lambdaas_schema: LambdaasSchema
    norm_operator: NormOperatorType
    linear_operator: LinearOperatorMethod
    nb_iters: int

    @property
    def prox_functional(self) -> CTVOperator:
        return CTVOperator(norm=NormOperator.from_norm(norm=self.norm_operator))

    @property
    def domain_transform(self) -> LinearOperator:
        return LinearOperator.from_method(method=self.linear_operator)

    def ip_kwargs(self, lambdaa: float) -> dict:
        ip_kwargs = {
            self.lambdaas_schema.key.value: lambdaa,
            "prox_functional": self.prox_functional,
            "domain_transform": self.domain_transform,
            "nb_iters": self.nb_iters,
        }
        return ip_kwargs


class InversionProtocolExperimentListSchema(Sequence, RootModel):
    root: List[InversionProtocolExperimentSchema]

    def __getitem__(self, item) -> InversionProtocolExperimentSchema:
        return self.root[item]

    def __len__(self) -> int:
        return len(self.root)
