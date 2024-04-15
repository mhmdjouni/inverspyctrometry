from dataclasses import dataclass, replace

import numpy as np
from matplotlib import pyplot as plt

from src.common_utils.custom_vars import Wvn, Acq
from src.common_utils.light_wave import Spectrum
from src.direct_model.interferometer import FabryPerotInterferometer
from src.inverse_model.inverspectrometer import FabryPerotInverSpectrometerHaar


@dataclass
class GaussianGenerator:
    coefficients: np.ndarray[tuple[int, Acq], np.dtype[np.float_]]
    means: np.ndarray[tuple[int, Acq], np.dtype[np.float_]]
    stds: np.ndarray[tuple[int, Acq], np.dtype[np.float_]]

    def min_max_parameters(
            self,
            ref_min,
            ref_max,
            new_min,
            new_max,
    ):
        means = new_min + (self.means - ref_min) / ref_max * (new_max - new_min)
        stds = self.stds / ref_max * (new_max - new_min)
        return replace(self, means=means, stds=stds)

    def generate(
            self,
            variable: np.ndarray[tuple[Wvn], np.dtype[np.float_]],
    ) -> np.ndarray[tuple[Wvn, Acq], np.dtype[np.float_]]:
        variable_centered = (variable[None, None, :] - self.means[:, :, None]) / self.stds[:, :, None]
        gaussian_funcs = np.exp(-variable_centered**2)
        data = np.sum(self.coefficients[:, :, None] * gaussian_funcs, axis=0).T
        return data


def main():
    wn_max_target = 1 / 350 * 1000  # um
    opd_step = 1 / (2 * wn_max_target)
    opd_num = 319
    opds = opd_step * np.arange(opd_num)

    wn_min = 1/1000 * 1000
    wn_max = 1/350 * 1000
    wn_num = 319
    wavenumbers = np.linspace(start=wn_min, stop=wn_max, num=wn_num)  # um-1
    paper_gaussian = GaussianGenerator(
        coefficients=np.array([1., 0.9, 0.75])[:, None],
        means=np.array([2., 4.25, 6.5])[:, None],
        stds=np.array([0.3, 1.125, 0.4])[:, None],
    )
    gaussian_gen = paper_gaussian.min_max_parameters(ref_min=0., ref_max=20., new_min=wn_min, new_max=wn_max)
    spectrum_data = gaussian_gen.generate(variable=wavenumbers)
    spectrum_ref = Spectrum(data=spectrum_data, wavenumbers=wavenumbers)

    reflectance = np.array([0.1])
    transmittance = np.array([1.])  # The values in the paper seem to be normalized by the transmittance

    fp_0 = FabryPerotInterferometer(
        transmittance_coefficients=transmittance,
        opds=opds,
        phase_shift=np.array([0]),
        reflectance_coefficients=reflectance,
        order=0,
    )
    interferogram = fp_0.acquire_interferogram(spectrum=spectrum_ref)

    order = 50
    fp_haar = FabryPerotInverSpectrometerHaar(
        transmittance=transmittance,
        wavenumbers=wavenumbers,
        reflectance=reflectance,
        order=order,
        is_mean_center=True,
    )

    coefficients = fp_haar.kernel_fourier_coefficients()
    spectrum_rec = fp_haar.reconstruct_spectrum(interferogram=interferogram)

    print(np.around(coefficients, decimals=3))
    print()

    acq_ind = 0

    fig, axs = plt.subplots(squeeze=False)
    spectrum_ref.visualize(axs=axs[0, 0], acq_ind=acq_ind)

    fig, axs = plt.subplots(squeeze=False)
    interferogram.visualize(axs=axs[0, 0], acq_ind=acq_ind)

    fig, axs = plt.subplots(squeeze=False)
    spectrum_rec_eq, spectrum_ref = spectrum_rec.match_stats(reference=spectrum_ref, axis=-2)
    spectrum_ref.visualize(axs=axs[0, 0], acq_ind=acq_ind, color="C0")
    spectrum_rec_eq.visualize(axs=axs[0, 0], acq_ind=acq_ind, linestyle="--", color="C1")
    plt.show()


if __name__ == "__main__":
    main()
