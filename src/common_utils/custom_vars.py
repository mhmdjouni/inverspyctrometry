from enum import Enum
from typing import NewType


Wvn = NewType(name='Wvn', tp=int)
Opd = NewType(name='Opd', tp=int)


class InterferometerType(str, Enum):
    MICHELSON = "Michelson"
    FABRY_PEROT = "Fabry-Perot"


class InversionProtocolType(str, Enum):
    IDCT = "idct"  # Inverse Discrete Cosine Transform
    PSEUDO_INVERSE = "pseudo_inverse"
    TSVD = "truncated_svd"  # Truncated Singular Value Decomposition
    RIDGE_REGRESSION = "ridge_regression"
    LORIS_VERHOEVEN = "loris_verhoeven"
    ADMM = "admm"  # Alternating Direction Method of Multipliers
