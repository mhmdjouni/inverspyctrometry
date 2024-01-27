from typing import List

import numpy as np
from pydantic import BaseModel

from src.common_utils.light_wave import Spectrum
from src.database.datasets import DatasetListSchema
from src.database.interferometers import InterferometerListSchema
from src.database.inversion_protocols import InversionProtocolListSchema


# TODO: Replace the type hints with Enums
# TODO: The lambdaas field in InversionProtocol should be just a float?


class DatabaseSchema(BaseModel):
    datasets: DatasetListSchema
    interferometers: InterferometerListSchema
    inversion_protocols: InversionProtocolListSchema
    noise_levels: List[float]
