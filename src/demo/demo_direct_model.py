import matplotlib.pyplot as plt
import numpy as np

from src.common_utils.custom_vars import InterferometerType as IfmType
from src.direct_model.interferometer import interferometer_factory
from src.common_utils.light_wave import Spectrum


def main():
    """
    For a future slider, my variables are:
    - Reflectance: To increase N as the N-wave approximation in FP with order 0
    - OPD index in the plot of the transfer matrices: To observe the OPD responses and their DCT
    - OPD value in the cosines that constitute the spectral radiance: Taken from the list of OPDs for controlled observation of the interferograms
    """
    nb_opd, del_opd = 320, 0.175
    opds = del_opd * np.arange(nb_opd)
    nb_wn = opds.size*6  # quasi-continuous
    del_wn = 1 / (2 * nb_wn * del_opd)  # tends to zero as nb_wn tends to infinity (implies continuous)
    wavenumbers = del_wn * (np.arange(nb_wn) + 1/2)

    reflectance = 0.13 * np.ones_like(a=wavenumbers, dtype=np.float_)
    transmittance = 1 - reflectance

    plot_opd_idx = 300
    radiance_cosine_args = {
        "amplitudes": np.array([1, 1]),
        "opds": opds[[50, 300]],
    }

    michelson = interferometer_factory(option=IfmType.MICHELSON, transmittance=transmittance, opds=opds)
    fabry_perot = interferometer_factory(option=IfmType.FABRY_PEROT, transmittance=transmittance, opds=opds)
    fabry_perot_2 = interferometer_factory(option=IfmType.FABRY_PEROT, transmittance=transmittance, opds=opds, order=2)

    transfer_matrix_mich = michelson.generate_transmittance_response(wavenumbers=wavenumbers)
    transfer_matrix_fp = fabry_perot.generate_transmittance_response(wavenumbers=wavenumbers)
    transfer_matrix_fp_2 = fabry_perot_2.generate_transmittance_response(wavenumbers=wavenumbers)

    spectrum = Spectrum.from_oscillations(
        amplitudes=radiance_cosine_args["amplitudes"],
        opds=radiance_cosine_args["opds"],
        wavenumbers=wavenumbers,
    )

    interferogram_mich = michelson.acquire_interferogram(spectrum=spectrum)
    interferogram_fp = fabry_perot.acquire_interferogram(spectrum=spectrum)
    interferogram_fp_2 = fabry_perot_2.acquire_interferogram(spectrum=spectrum)

    fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False)
    transfer_matrix_mich.visualize(axs=axs[0, 0])
    transfer_matrix_mich.visualize_opd_response(axs=axs[0, 1], opd_idx=plot_opd_idx)
    transfer_matrix_mich.visualize_dct(axs=axs[1, 0], opd_idx=-1)
    transfer_matrix_mich.visualize_singular_values(axs=axs[1, 1])

    fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False)
    transfer_matrix_fp_2.visualize(axs=axs[0, 0])
    transfer_matrix_fp_2.visualize_opd_response(axs=axs[0, 1], opd_idx=plot_opd_idx)
    transfer_matrix_fp_2.visualize_dct(axs=axs[1, 0], opd_idx=-1)
    transfer_matrix_fp_2.visualize_singular_values(axs=axs[1, 1])

    fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False)
    transfer_matrix_fp.visualize(axs=axs[0, 0])
    transfer_matrix_fp.visualize_opd_response(axs=axs[0, 1], opd_idx=plot_opd_idx)
    transfer_matrix_fp.visualize_dct(axs=axs[1, 0], opd_idx=-1)
    transfer_matrix_fp.visualize_singular_values(axs=axs[1, 1])

    fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False)
    spectrum.visualize(axs=axs[0, 0])
    interferogram_mich.visualize(axs=axs[0, 1])
    interferogram_fp.visualize(axs=axs[1, 0])
    interferogram_fp_2.visualize(axs=axs[1, 1])

    plt.show()


if __name__ == "__main__":
    main()
