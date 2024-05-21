from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from src.common_utils.function_generator import GaussianGenerator
from src.common_utils.light_wave import Spectrum
from src.direct_model.interferometer import FabryPerotInterferometer
from src.inverse_model.inverspectrometer import calculate_airy_fourier_coeffs


def compute_wavenumbers(opd, wn, wn_num_factor):
    opd_max = opd.num * opd.step
    wn_step_min = 1 / (2 * opd_max)  # Nyquist-Shannon bandwidth
    wn_step = wn_step_min / wn_num_factor  # oversampling to mimic continuity
    wavenumbers = np.arange(wn.start, wn.stop, wn_step)
    return wavenumbers


def oversample_wavenumbers(wavenumbers, factor):
    original_points = len(wavenumbers)
    new_points = original_points * factor
    original_indices = np.arange(original_points)
    new_indices = np.linspace(0, original_points - 1, new_points)
    interp_func = interp1d(original_indices, wavenumbers, kind='linear')
    oversampled_wavenumbers = interp_func(new_indices)
    return oversampled_wavenumbers


def generate_synthetic_spectrum(gauss, opd, wn):
    opd_max = opd.num * opd.step
    wn_step = 1 / (2 * opd_max)  # Nyquist-Shannon bandwidth

    wavenumbers = np.arange(wn.start, wn.stop, wn_step)
    gaussian_gen = GaussianGenerator(
        coefficients=gauss.coeffs[:, None],
        means=gauss.means[:, None],
        stds=gauss.stds[:, None],
    )
    spectrum_data = gaussian_gen.generate_data(variable=wavenumbers)
    spectrum_ref = Spectrum(data=spectrum_data, wavenumbers=wavenumbers, wavenumbers_unit=wn.unit)

    return spectrum_ref


def oversample_spectrum(spectrum_reference, opd_info, wn_continuity_factor):
    wn_step = np.mean(np.diff(spectrum_reference.wavenumbers))
    opd_info_max = opd_info.step * (opd_info.num - 1)
    wn_step_nyquist = 1 / (2 * opd_info_max)
    wn_step_target = wn_step_nyquist / wn_continuity_factor
    wn_factor_new = int(np.ceil(wn_step / wn_step_target))
    wavenumbers_new = oversample_wavenumbers(spectrum_reference.wavenumbers, factor=wn_factor_new)
    spectrum_continuous = spectrum_reference.interpolate(wavenumbers=wavenumbers_new, kind="slinear")
    return spectrum_continuous


def generate_interferogram(opd_info, fp, spectrum_reference, wn_continuity_factor):
    opds = np.arange(0, opd_info.step * opd_info.num, opd_info.step)
    device = FabryPerotInterferometer(
        transmittance_coefficients=fp.transmittance,
        opds=opds,
        phase_shift=fp.phase_shift,
        reflectance_coefficients=fp.reflectance,
        order=fp.order,
    )

    spectrum_continuous = oversample_spectrum(spectrum_reference, opd_info, wn_continuity_factor)
    interferogram = device.acquire_interferogram(spectrum_continuous)

    wn_step = np.mean(np.diff(spectrum_continuous.wavenumbers))
    interferogram = replace(interferogram, data=interferogram.data * wn_step, opds_unit=opd_info.unit)

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
