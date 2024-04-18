from dataclasses import replace

import numpy as np
from matplotlib import pyplot as plt

from src.common_utils.light_wave import Spectrum
from src.common_utils.function_generator import GaussianGenerator
from src.direct_model.interferometer import FabryPerotInterferometer
from src.inverse_model.inverspectrometer import FabryPerotInverSpectrometerHaar
from src.inverse_model.protocols import IDCT, PseudoInverse


def main():
    is_imshow = True
    for fp_order in [0]:
        for reflectance in [0.7]:
            for haar_order in [25]:
                (
                    interferogram,
                    spectrum_ref,
                    spectrum_idct,
                    spectrum_haar,
                ) = paper_test(
                    gauss_coeffs=np.array([1., 0.9, 0.75]),
                    gauss_means=np.array([2000, 4250, 6500]),  # cm
                    gauss_stds=np.array([300, 1125, 400]),  # cm
                    opd_step=100*1e-7,  # cm
                    opd_num=2048,
                    reflectance=np.array([reflectance]),
                    # wn_min=24.31 * 1,  # cm
                    # wn_max=24.41 * 7,  # cm
                    wn_min=20.,  # cm
                    wn_max=20000.1,  # cm
                    haar_order=haar_order,
                    fp_order=fp_order,
                    wn_num_factor=10,
                    idct_correction=1.4,
                )

                if is_imshow:
                    visualize_test(
                        reflectance=reflectance,
                        haar_order=haar_order,
                        interferogram=interferogram,
                        spectrum_ref=spectrum_ref,
                        spectrum_idct=spectrum_idct,
                        spectrum_haar=spectrum_haar,
                        acq_ind=0,
                        ylim=[-0.2, 1.5],
                    )
        plt.show()


def paper_test(
        gauss_coeffs,
        gauss_means,
        gauss_stds,
        opd_step: float,
        opd_num: int,
        reflectance,
        wn_min: float,
        wn_max: float,
        haar_order: int,
        fp_order: int,
        wn_num_factor: float,
        idct_correction: float,
):
    opds = opd_step * np.arange(opd_num)

    # wn_step = 1 / (2 * opds.max())
    wn_max_dct = 1 / (2 * opd_step)
    wn_num = int(opd_num * (wn_max - wn_min) / wn_max_dct * wn_num_factor)
    wavenumbers = np.linspace(start=wn_min, stop=wn_max, num=wn_num, endpoint=False)  # um-1

    gaussian_gen = GaussianGenerator(
        coefficients=gauss_coeffs[:, None],
        means=gauss_means[:, None],
        stds=gauss_stds[:, None],
    )
    spectrum_data = gaussian_gen.generate_data(variable=wavenumbers)
    spectrum_ref = Spectrum(data=spectrum_data, wavenumbers=wavenumbers)

    transmittance = np.array([1.])  # The values in the paper seem to be normalized by the transmittance

    fp_0 = FabryPerotInterferometer(
        transmittance_coefficients=transmittance,
        opds=opds,
        phase_shift=np.array([0]),
        reflectance_coefficients=reflectance,
        order=fp_order,
    )
    interferogram = fp_0.acquire_interferogram(spectrum=spectrum_ref)
    interferogram = replace(interferogram, data=interferogram.data / wn_num_factor / idct_correction)

    wn_step = 1 / (2 * opds.max())
    wn_max_dct = 1 / (2 * opd_step)
    wn_num = int(opd_num * (wn_max - wn_min) / wn_max_dct)
    wavenumbers = wn_min + wn_step * np.arange(wn_num)  # um-1

    fp_haar = FabryPerotInverSpectrometerHaar(
        transmittance=transmittance,
        wavenumbers=wavenumbers,
        reflectance=reflectance,
        order=haar_order,
        is_mean_center=True,
    )
    spectrum_haar = fp_haar.reconstruct_spectrum(interferogram=interferogram)

    idct_inv = IDCT(is_mean_center=True)
    transmittance_response = fp_0.transmittance_response(wavenumbers=wavenumbers, is_correct_transmittance=False)
    spectrum_idct = idct_inv.reconstruct_spectrum(
        interferogram=interferogram,
        transmittance_response=transmittance_response
    )

    return (
        interferogram,
        spectrum_ref,
        spectrum_idct,
        spectrum_haar,
    )


def visualize_test(
        reflectance,
        haar_order,
        interferogram,
        spectrum_ref,
        spectrum_idct,
        spectrum_haar,
        acq_ind,
        ylim,
):
    fig, axs = plt.subplots(1, 2, squeeze=False, figsize=(9, 5))

    axs_current = axs[0, 0]
    interferogram.visualize(axs=axs_current, acq_ind=acq_ind)
    axs_current.set_title(f'Simulated Interferogram, R = {reflectance}')
    axs_current.set_xlabel('OPDs [cm]')
    axs_current.set_ylabel('Intensity')

    axs_current = axs[0, 1]
    spectrum_ref.visualize(axs=axs_current, acq_ind=acq_ind, color="red", label='Reference', ylim=ylim)
    spectrum_idct_eq, _ = spectrum_idct.match_stats(reference=spectrum_ref, axis=-2)
    spectrum_idct.visualize(axs=axs_current, acq_ind=acq_ind, linestyle="dashed", color="green", label='IDCT', ylim=ylim)
    spectrum_haar_eq, _ = spectrum_haar.match_stats(reference=spectrum_ref, axis=-2)
    spectrum_haar.visualize(axs=axs_current, acq_ind=acq_ind, linestyle="dashed", color="blue", label='Haar', ylim=ylim)
    axs_current.set_title(f'Reconstructed Spectrum, M = {haar_order}')
    axs_current.set_xlabel('Wavenumbers [cm-1]')
    axs_current.set_ylabel('Intensity')

    plt.legend()
    plt.grid(True)


if __name__ == "__main__":
    main()
