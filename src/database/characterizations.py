from typing import Sequence

import numpy as np
from pydantic import BaseModel, FilePath, RootModel

from src.common_utils.custom_vars import DeviceType, InterferometerType
from src.direct_model.characterization import Characterization


class CharacterizationSchema(BaseModel):
    id: int
    title: str
    device: DeviceType
    category: str
    source_dataset_id: int
    interferometer_type: InterferometerType
    order: int
    opds: FilePath
    transmittance_coefficients: FilePath
    reflectance_coefficients: FilePath
    phase_shift: FilePath

    def characterization(self) -> Characterization:
        transmittance_coefficients = np.load(self.transmittance_coefficients)
        reflectance_coefficients = np.load(self.reflectance_coefficients)
        opds = np.load(self.opds)
        phase_shift = np.load(self.phase_shift)

        if phase_shift.size == 1:
            phase_shift = np.tile(phase_shift, self.opds.num)

        return Characterization(
            interferometer_type=self.interferometer_type,
            transmittance_coefficients=transmittance_coefficients,
            opds=opds,
            phase_shift=phase_shift,
            reflectance_coefficients=reflectance_coefficients,
            order=self.order,
        )


class CharacterizationListSchema(Sequence, RootModel):
    root: list[CharacterizationSchema]

    def __getitem__(self, item: int) -> CharacterizationSchema:
        return self.root[item]

    def __len__(self) -> int:
        return len(self.root)
