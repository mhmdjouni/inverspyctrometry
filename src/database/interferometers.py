from typing import Optional, Sequence, List

from pydantic import BaseModel, RootModel


# TODO: Separate OPD Schema and add OPD id instead


class OPDSchema(BaseModel):
    nb: int
    step: float
    min: float


class InterferometerSchema(BaseModel):
    # TODO: Add category (experiment) and experiment id
    id: int
    title: str
    type: str
    opds: OPDSchema
    transmittance: float
    reflectance: Optional[float] = None
    order: Optional[int] = None


class InterferometerListSchema(Sequence, RootModel):
    root: List[InterferometerSchema]

    def __getitem__(self, item: int) -> InterferometerSchema:
        return self.root[item]

    def __len__(self) -> int:
        return len(self.root)
