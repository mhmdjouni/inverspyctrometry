from typing import Optional, Sequence, List

from pydantic import BaseModel, RootModel


class LambdaasSchema(BaseModel):
    min: float
    max: float
    nb: int
    space: str


class InversionProtocolSchema(BaseModel):
    id: int
    title: str
    type: str
    lambdaas: Optional[LambdaasSchema] = None
    norm_operator: Optional[str] = None
    linear_operator: Optional[str] = None
    nb_iters: Optional[int] = None


class InversionProtocolListSchema(Sequence, RootModel):
    root: List[InversionProtocolSchema]

    def __getitem__(self, item) -> InversionProtocolSchema:
        return self.root[item]

    def __len__(self) -> int:
        return len(self.root)
