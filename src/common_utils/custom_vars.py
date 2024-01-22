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


class LinearOperatorMethod(str, Enum):
    IDENTITY = "identity"
    DCT = "dct"  # Discrete Cosine Transform
    TV = "tv"  # Total Variation


class NormOperatorType(tuple, Enum):
    L1O = ("l1o", ())
    L2O = ("l2o", ())
    L112 = ("l112", (2, 0, 1))
    L121 = ("l121", (1, 0, 2))
    L211 = ("l211", (0, 1, 2))
