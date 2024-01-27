from pathlib import Path
from typing import Sequence, List, Optional

import numpy as np
from pydantic import BaseModel, RootModel, FilePath

from src.common_utils.light_wave import Spectrum


# TODO: Replace the type hints with Enums


class DatasetSchema(BaseModel):
    id: int
    title: str
    category: str
    device: str
    path: Path
    wavenumbers_path: Optional[Path] = None
    wavenumbers_unit: Optional[str] = None
    opds_path: Optional[Path] = None
    opds_unit: Optional[str] = None


class DatasetListSchema(Sequence, RootModel):
    root: List[DatasetSchema]

    def __getitem__(self, item) -> DatasetSchema:
        return self.root[item]

    def __len__(self) -> int:
        return len(self.root)
