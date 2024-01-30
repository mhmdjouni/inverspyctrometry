from typing import List

import numpy as np
from pydantic import BaseModel

from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum
from src.database.datasets import DatasetListSchema, DatasetSchema
from src.database.interferometers import InterferometerListSchema, InterferometerSchema
from src.database.inversion_protocols import InversionProtocolExperimentListSchema, InversionProtocolExperimentSchema
from src.direct_model.interferometer import Interferometer


class DatabaseSchema(BaseModel):
    datasets: list[DatasetSchema]
    interferometers: list[InterferometerSchema]
    inversion_protocols: list[InversionProtocolExperimentSchema]
    noise_levels: list[float]

    def dataset_spectrum(self, ds_id: int) -> Spectrum:
        return self.datasets[ds_id].spectrum()

    def dataset_interferogram(self, ds_id: int) -> Interferogram:
        return self.datasets[ds_id].interferogram()

    def interferometer(self, ifm_id: int) -> Interferometer:
        return self.interferometers[ifm_id].interferometer()
