import numpy as np
from matplotlib import pyplot as plt

from src.common_utils.light_wave import Spectrum
from src.common_utils.function_generator import GaussianGenerator
from src.direct_model.interferometer import FabryPerotInterferometer
from src.inverse_model.inverspectrometer import FabryPerotInverSpectrometerHaar


def my_test():
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


def paper_test(
        gauss_coeffs,
        gauss_means,
        gauss_stds,
        reflectance,
        opd_step: float,
        opd_num: int,
        wn_min: float,
        wn_max: float,
):
    opds = opd_step * np.arange(opd_num) + opd_step

    wn_step = 1 / (2 * opds.max())
    wn_max_dct = 1 / (2 * opd_step)
    wn_num = int(opd_num * (wn_max - wn_min) / wn_max_dct)
    wavenumbers = wn_min + wn_step * np.arange(wn_num)  # um-1
    gaussian_gen = GaussianGenerator(
        coefficients=gauss_coeffs[:, None],
        means=gauss_means[:, None],
        stds=gauss_stds[:, None],
    )
    spectrum_data = gaussian_gen.generate(variable=wavenumbers)
    spectrum_ref = Spectrum(data=spectrum_data, wavenumbers=wavenumbers)

    transmittance = np.array([1.])  # The values in the paper seem to be normalized by the transmittance

    fp_0 = FabryPerotInterferometer(
        transmittance_coefficients=transmittance,
        opds=opds,
        phase_shift=np.array([0]),
        reflectance_coefficients=reflectance,
        order=0,
    )
    interferogram = fp_0.acquire_interferogram(spectrum=spectrum_ref)

    order = 20
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

    fig, axs = plt.subplots(1, 3, squeeze=False)

    spectrum_ref.visualize(axs=axs[0, 0], acq_ind=acq_ind)

    interferogram.visualize(axs=axs[0, 1], acq_ind=acq_ind)

    spectrum_rec_eq, spectrum_ref = spectrum_rec.match_stats(reference=spectrum_ref, axis=-2)
    spectrum_ref.visualize(axs=axs[0, 2], acq_ind=acq_ind, color="C0")
    spectrum_rec_eq.visualize(axs=axs[0, 2], acq_ind=acq_ind, linestyle="--", color="C1")

    plt.show()


def main():
    paper_test(
        gauss_coeffs=np.array([1., 0.9, 0.75]),
        gauss_means=np.array([2000, 4250, 6500]),  # cm
        gauss_stds=np.array([300, 1125, 400]),  # cm
        reflectance=np.array([0.7]),
        opd_step=100*1e-7,  # cm
        opd_num=2048,
        wn_min=1000.,  # cm
        wn_max=8000.,  # cm
    )


if __name__ == "__main__":
    main()
