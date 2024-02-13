from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlotOptions:
    subplots_options: SubplotsOptions
    rc_params: RcParamsOptions


@dataclass
class RcParamsOptions:
    fontsize: int = 10


@dataclass
class SubplotsOptions:
    nrows: int = 1
    ncols: int = 1
    squeeze: bool = False
    figsize: tuple = (6.4, 4.8)
    tight_layout: bool = True
