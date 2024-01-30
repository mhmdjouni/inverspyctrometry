from typing import Optional, Sequence, List

import numpy as np
from pydantic import BaseModel, RootModel

from src.common_utils.custom_vars import InterferometerType, Opd
from src.common_utils.utils import generate_sampled_opds
from src.direct_model.interferometer import Interferometer, interferometer_factory


# TODO: Separate OPD Schema and add OPD id instead


class OPDSchema(BaseModel):
    num: int
    step: float
    start: float

    def generate_opds(self) -> np.ndarray[tuple[Opd], np.dtype[np.float_]]:
        return generate_sampled_opds(nb_opd=self.num, opd_step=self.step, opd_min=self.start)


class InterferometerSchema(BaseModel):
    # TODO: Add category (experiment) and experiment id
    id: int
    title: str
    category: str
    type: InterferometerType
    opds: OPDSchema
    transmittance: list[float]
    reflectance: Optional[list[float]] = None
    order: Optional[int] = None

    def interferometer(self) -> Interferometer:
        transmittance = np.array(self.transmittance)
        reflectance = np.array(self.reflectance)
        opds = self.opds.generate_opds()
        interferometer = interferometer_factory(
            option=self.type,
            transmittance=transmittance,
            opds=opds,
            reflectance=reflectance,
            order=self.order,
        )
        return interferometer


class InterferometerListSchema(Sequence, RootModel):
    root: List[InterferometerSchema]

    def __getitem__(self, item: int) -> InterferometerSchema:
        return self.root[item]

    def __len__(self) -> int:
        return len(self.root)
