from dataclasses import replace
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from src.common_utils.function_generator import GaussianGenerator
from src.common_utils.light_wave import Spectrum
from src.direct_model.interferometer import FabryPerotInterferometer
from src.inverse_model.inverspectrometer import compute_interferogram_dft, calculate_airy_fourier_coeffs


def generate_synthetic_spectrum(opd, wn, gauss):
    opd_max = opd.num * opd.step
    wn_step_min = 1 / (2 * opd_max)  # Nyquist-Shannon bandwidth
    wn_step = wn_step_min / wn.num_factor  # oversampling to mimic continuity

    wavenumbers = np.arange(wn.start, wn.stop, wn_step)
    gaussian_gen = GaussianGenerator(
        coefficients=gauss.coeffs[:, None],
        means=gauss.means[:, None],
        stds=gauss.stds[:, None],
    )
    spectrum_data = gaussian_gen.generate_data(variable=wavenumbers)
    spectrum_ref = Spectrum(data=spectrum_data, wavenumbers=wavenumbers, wavenumbers_unit=wn.unit)

    return spectrum_ref


def generate_interferogram(opd, fp, spectrum_cont):
    opds = np.arange(0, opd.step * opd.num, opd.step)

    device = FabryPerotInterferometer(
        transmittance_coefficients=fp.transmittance,
        opds=opds,
        phase_shift=fp.phase_shift,
        reflectance_coefficients=fp.reflectance,
        order=fp.order,
    )
    interferogram = device.acquire_interferogram(spectrum_cont)

    wn_step = np.mean(np.diff(spectrum_cont.wavenumbers))
    interferogram = replace(interferogram, data=interferogram.data * wn_step, opds_unit=opd.unit)

    return interferogram


def assert_haar_check(fp, interferogram_dft, spectrum, haar_order):
    k_vals = np.where(
        np.logical_and(
            spectrum.wavenumbers[0] <= interferogram_dft.wavenumbers,
            interferogram_dft.wavenumbers <= spectrum.wavenumbers[-1],
        )
    )

    a_cap = calculate_airy_fourier_coeffs(fp, haar_order)

    k_cap = interferogram_dft.wavenumbers.size
    b_matrix = np.zeros(shape=(k_cap, k_cap))
    for mu in k_vals[0]:
        for n in range(1, haar_order + 1):
            for k in k_vals[0]:
                for i in range(n):
                    if n*k+i == mu:
                        b_matrix[mu, k] += a_cap[n] / n

    # spectrum_downsamp = replace(spectrum, data=spectrum.data[::20], wavenumbers=spectrum.wavenumbers[::20])
    spectrum_interp = spectrum.interpolate(wavenumbers=interferogram_dft.wavenumbers, kind="cubic")

    # fig, axs = plt.subplots()
    # spectrum_interp.visualize(axs=axs, acq_ind=0, color="C0", label="interpolated")
    # spectrum_downsamp.visualize(axs=axs, acq_ind=0, color="C1", label="downsampled")
    # spectrum.visualize(axs=axs, acq_ind=0, color="C2", label="reference")
    # plt.show()

    spectrum_transform = replace(spectrum_interp, data=b_matrix @ spectrum_interp.data)

    return spectrum_transform, b_matrix


def main():
    opd_info_obj = SimpleNamespace(
        step=100 * 1e-7,  # 100 nm => cm
        num=2048,
        unit="cm",
    )
    wn_bounds_obj = SimpleNamespace(
        start=0.,  # cm-1
        stop=20000.1,  # cm-1
        num_factor=10,
        unit="1/cm",
    )  # nm
    gauss_params_obj = SimpleNamespace(
        coeffs=np.array([1., 0.9, 0.75]),
        means=np.array([2000, 4250, 6500]),  # cm-1
        stds=np.array([300, 1125, 400]),  # cm-1
    )
    fp_obj = SimpleNamespace(
        transmittance=np.array([1.]),
        phase_shift=np.array([0.]),
        reflectance=np.array([0.7]),
        order=0,
    )

    spectrum_ref = generate_synthetic_spectrum(opd_info_obj, wn_bounds_obj, gauss_params_obj)

    interferogram_sim = generate_interferogram(opd_info_obj, fp_obj, spectrum_ref)

    acq_idx = 0
    fig, axs = plt.subplots(1, 2)
    axs_spc, axs_ifm = axs
    spectrum_ref.visualize(axs=axs_spc, acq_ind=acq_idx)
    interferogram_sim.visualize(axs=axs_ifm, acq_ind=acq_idx)

    interferogram_dft = compute_interferogram_dft(interferogram_sim)

    spectrum_transform, b_matrix = assert_haar_check(fp_obj, interferogram_dft, spectrum_ref, haar_order=10)

    acq_idx = 0
    fig, axs = plt.subplots(1, 2)
    axs_ifm, axs_spc = axs
    interferogram_dft.visualize(axs=axs_ifm, acq_ind=acq_idx, title="Interferogram DFT " + r"[$J_u$]")
    spectrum_transform.visualize(axs=axs_spc, acq_ind=acq_idx, title="Spectrum transform " + r"$B$[$a_k$]")

    spectrum_rec = replace(interferogram_dft, data=np.linalg.pinv(b_matrix) @ interferogram_dft.data)
    acq_idx = 0
    fig, axs = plt.subplots(1, 1)
    spectrum_rec.visualize(axs=axs, acq_ind=acq_idx, title="Spectrum transform " + r"[$a_k$]")

    plt.show()


if __name__ == "__main__":
    main()
