from typing import List

import numpy as np
from pydantic import BaseModel

from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum
from src.database.characterizations import CharacterizationSchema
from src.database.datasets import DatasetListSchema, DatasetSchema
from src.database.experiments import ExperimentSchema
from src.database.interferometers import InterferometerListSchema, InterferometerSchema
from src.database.inversion_protocols import InversionProtocolListSchema, InversionProtocolSchema
from src.direct_model.characterization import Characterization
from src.direct_model.interferometer import Interferometer


class DatabaseSchema(BaseModel):
    characterizations: list[CharacterizationSchema]
    datasets: list[DatasetSchema]
    experiments: list[ExperimentSchema]
    interferometers: list[InterferometerSchema]
    inversion_protocols: list[InversionProtocolSchema]
    noise_levels: list[float]

    def characterization(self, char_id: int) -> Characterization:
        return self.characterizations[char_id].characterization()

    def dataset_spectrum(self, ds_id: int) -> Spectrum:
        return self.datasets[ds_id].spectrum()

    def dataset_interferogram(self, ds_id: int) -> Interferogram:
        return self.datasets[ds_id].interferogram()

    def interferometer(self, ifm_id: int) -> Interferometer:
        return self.interferometers[ifm_id].interferometer()
