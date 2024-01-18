from enum import Enum
from typing import NewType


Wvn = NewType(name='Wvn', tp=int)
Opd = NewType(name='Opd', tp=int)


class InterferometerType(str, Enum):
    MICHELSON = "Michelson"
    FABRY_PEROT = "Fabry-Perot"
