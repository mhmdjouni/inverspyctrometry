from typing import List

import numpy as np
from pydantic import BaseModel

from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum
from src.database.datasets import DatasetListSchema
from src.database.interferometers import InterferometerListSchema
from src.database.inversion_protocols import InversionProtocolListSchema


class DatabaseSchema(BaseModel):
    datasets: DatasetListSchema
    interferometers: InterferometerListSchema
    inversion_protocols: InversionProtocolListSchema
    noise_levels: List[float]

    def dataset_spectrum(self, ds_id: int) -> Spectrum:
        return self.datasets[ds_id].spectrum()

    def dataset_interferogram(self, ds_id: int) -> Interferogram:
        return self.datasets[ds_id].interferogram()
