"""
Reproducing the paper's restults
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from matplotlib import pyplot as plt

from src.common_utils.interferogram import Interferogram
from src.demo.paper.haar.utils import generate_synthetic_spectrum, generate_interferogram
from src.direct_model.interferometer import FabryPerotInterferometer
from src.inverse_model.inverspectrometer import FabryPerotInverSpectrometerHaar
from src.inverse_model.protocols import IDCT


def invert_haar(wavenumbers, fp, haar_order, interferogram: Interferogram):
    haar = FabryPerotInverSpectrometerHaar(
        transmittance=fp.transmittance,
        wavenumbers=wavenumbers,
        reflectance=fp.reflectance,
        order=haar_order,
        is_mean_center=True,
    )
    spectrum = haar.reconstruct_spectrum(interferogram=interferogram)
    return spectrum


def invert_idct(wavenumbers, fp, interferogram: Interferogram):
    device = FabryPerotInterferometer(
        transmittance_coefficients=fp.transmittance,
        opds=interferogram.opds,
        phase_shift=fp.phase_shift,
        reflectance_coefficients=fp.reflectance,
        order=fp.order,
    )
    transmittance_response = device.transmittance_response(wavenumbers=wavenumbers)
    idct = IDCT(is_mean_center=True)
    spectrum = idct.reconstruct_spectrum(interferogram=interferogram, transmittance_response=transmittance_response)
    return spectrum


def compute_wavenumbers(opd, wn):
    opd_max = opd.num * opd.step
    wn_step_min = 1 / (2 * opd_max)  # Nyquist-Shannon bandwidth
    wn_step = wn_step_min / wn.num_factor  # oversampling to mimic continuity
    wavenumbers = np.arange(wn.start, wn.stop, wn_step)
    return wavenumbers


def paper_test_modified():
    # OPTIONS

    opd_info_obj = SimpleNamespace(
        step=100 * 1e-7,  # 100 nm => cm
        num=2048,
        unit="cm",
    )
    wn_bounds_obj = SimpleNamespace(
        start=0,  # cm-1
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
    haar_order = 10
    snr_db = None

    # SIMULATION

    spectrum_ref = generate_synthetic_spectrum(opd_info_obj, wn_bounds_obj, gauss_params_obj)

    # OBSERVATION

    interferogram_sim = generate_interferogram(opd_info_obj, fp_obj, spectrum_ref)
    interferogram_sim = interferogram_sim.center(new_mean=0., axis=-2)
    interferogram_sim = interferogram_sim.rescale(new_max=1., axis=-2)
    if snr_db is not None:
        interferogram_sim = interferogram_sim.add_noise(snr_db=snr_db)

    # INVERSION

    wavenumbers = compute_wavenumbers(opd_info_obj, wn_bounds_obj)
    spectrum_haar = invert_haar(wavenumbers, fp_obj, haar_order, interferogram_sim)
    spectrum_idct = invert_idct(wavenumbers, fp_obj, interferogram_sim)

    # VISUALIZATION

    acq_idx = 0
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    axs_spc, axs_ifm, axs_rec = axs
    spc_ylim = [-0.1, 1.4]

    spectrum_ref.visualize(
        axs=axs_spc,
        acq_ind=acq_idx,
        label="Reference",
        color="red",
        ylim=spc_ylim,
    )

    interferogram_sim.visualize(
        axs=axs_ifm,
        acq_ind=acq_idx,
        title="Simulated Interferogram",
        color="red",
    )

    spectrum_ref.visualize(
        axs=axs_rec,
        acq_ind=acq_idx,
        label="Reference",
        color="red",
        ylim=spc_ylim,
    )

    spectrum_protocol = replace(spectrum_idct, data=(spectrum_idct.data - 0.0066) / 0.083 * spectrum_ref.data.max())
    spectrum_protocol.visualize(
        axs=axs_rec,
        acq_ind=acq_idx,
        label="IDCT",
        color="green",
        linestyle="--",
        ylim=spc_ylim,
    )

    spectrum_haar = replace(spectrum_haar, data=(spectrum_haar.data - 0.0024) / 0.021 * spectrum_ref.data.max())
    spectrum_haar.visualize(
        axs=axs_rec,
        acq_ind=acq_idx,
        label="HAAR",
        color="blue",
        linestyle=":",
        marker="x",
        markevery=40,
        ylim=spc_ylim,
    )

    plt.show()


if __name__ == "__main__":
    paper_test_modified()
