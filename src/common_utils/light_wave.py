from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import interpolate

from src.common_utils.custom_vars import Wvn, Acq
from src.common_utils.utils import convert_meter_units


@dataclass(frozen=True)
class Spectrum:
    data: np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]]
    wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]]
    wavenumbers_unit: str = "1/cm"

    def visualize(self, axs, acq_ind: int):
        axs.plot(self.wavenumbers, self.data[:, acq_ind])
        axs.set_title(rf"Spectral Radiance")
        axs.set_ylabel("Intensity")
        axs.set_xlabel(rf"Wavenumbers $\sigma$ [{self.wavenumbers_unit}]")

    @classmethod
    def from_wavelength(
            cls,
            data: np.ndarray[tuple[int, Acq], np.dtype[np.float_]],
            wavelengths: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
            wavelengths_unit: str,
            is_interpolate: bool = False,
            interpolate_options: Optional[dict] = None,
            new_wavelength_unit: str = "cm",
    ) -> "Spectrum":
        unit_conversion_coefficient = convert_meter_units(values=1, from_=wavelengths_unit, to_=new_wavelength_unit)
        wavenumbers = 1 / wavelengths[::-1] / unit_conversion_coefficient
        spectrum = cls(
            data=data[::-1, :],
            wavenumbers=wavenumbers,
            wavenumbers_unit=f"1/{new_wavelength_unit}",
        )
        if is_interpolate:
            if interpolate_options is None:
                wavenumbers = np.linspace(start=wavenumbers[0], stop=wavenumbers[-1], num=wavenumbers.size)
            elif interpolate_options is not None and "wavenumbers" in interpolate_options:
                wavenumbers = interpolate_options["wavenumbers"] / unit_conversion_coefficient
            else:
                raise ValueError(f"The dictionary interpolate_options must either be {None} or contain the field 'wavenumbers'.")
            spectrum = spectrum.interpolate(wavenumbers=wavenumbers)
        return spectrum

    def interpolate(
            self,
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
            kind: str = "cubic",
    ) -> "Spectrum":
        spectrum_out = interpolate.interp1d(
            x=self.wavenumbers,
            y=self.data,
            axis=-2,
            kind=kind,
            fill_value=(0, 0),
            bounds_error=False,
        )(wavenumbers)
        return Spectrum(
            data=spectrum_out,
            wavenumbers=wavenumbers,
            wavenumbers_unit=self.wavenumbers_unit,
        )

    @classmethod
    def from_oscillations(
            cls,
            amplitudes: np.ndarray,
            opds: np.ndarray,
            wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> Spectrum:
        radiance = generate_radiance_from_oscillations(
            amplitudes=amplitudes,
            opds=opds,
            wavenumbers=wavenumbers,
        ).reshape((-1, 1))
        return Spectrum(data=radiance, wavenumbers=wavenumbers)


def generate_radiance_from_oscillations(
        amplitudes: np.ndarray,
        opds: np.ndarray,
        wavenumbers: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
) -> np.ndarray[tuple[Wvn], np.dtype[np.float_]]:
    if amplitudes.size != opds.size:
        raise ValueError("Amplitudes and OPDs must have the same size.")
    list_of_cosines = amplitudes[:, None] * np.cos(2 * np.pi * opds[:, None] * wavenumbers[None, :])
    radiance = np.sum(list_of_cosines, axis=0)
    return radiance


def convert_to_wavenumbers(
    spectrum_in_wavelengths: np.ndarray[tuple[int, Acq], np.dtype[np.float_]],
    wavelengths: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    wavelengths_unit: str = "nm",
    wavenumbers_limits: tuple[float, float] = None,
    wavenumbers_samples: int = None,
) -> Spectrum:
    spectrum_in = Spectrum.from_wavelength(
        data=spectrum_in_wavelengths,
        wavelengths=wavelengths,
        wavelengths_unit=wavelengths_unit,
    )

    if wavenumbers_limits is None:
        wavenumbers_limits = (1/wavelengths[-1], 1/wavelengths[0])
    if wavenumbers_samples is None:
        wavenumbers_samples = wavelengths.size
    wavenumbers = np.linspace(
        start=wavenumbers_limits[0],
        stop=wavenumbers_limits[1],
        num=wavenumbers_samples,
        endpoint=False,
    )
    spectrum_out = spectrum_in.interpolate(wavenumbers=wavenumbers)

    return spectrum_out
