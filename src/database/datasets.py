from pathlib import Path
from typing import Sequence, List, Optional

import numpy as np
from pydantic import BaseModel, RootModel, FilePath

from src.common_utils.custom_vars import DatasetCategory
from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum


# TODO: Replace the type hints with Enums
# TODO: Switch Path to FilePath


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

    def spectrum(self) -> Spectrum:
        if self.category != DatasetCategory.SPECTRUM:
            raise ValueError(f"The selected dataset id {self.id} does not refer to a spectral dataset.")
        data = np.load(file=self.path)
        wavenumbers = np.load(file=self.wavenumbers_path)
        wavenumbers_unit = self.wavenumbers_unit
        return Spectrum(
            data=data,
            wavenumbers=wavenumbers,
            wavenumbers_unit=wavenumbers_unit,
        )

    def interferogram(self) -> Interferogram:
        if self.category != DatasetCategory.INTERFEROGRAM:
            raise ValueError(f"The selected dataset id {self.id} does not refer to an interferogram dataset.")
        data = np.load(file=self.path)
        opds = np.load(file=self.opds_path)
        opds_unit = self.opds_unit
        return Interferogram(
            data=data,
            opds=opds,
            opds_unit=opds_unit,
        )


class DatasetListSchema(Sequence, RootModel):
    root: List[DatasetSchema]

    def __getitem__(self, item) -> DatasetSchema:
        return self.root[item]

    def __len__(self) -> int:
        return len(self.root)
