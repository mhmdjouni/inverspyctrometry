from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from matplotlib import pyplot as plt

from src.common_utils.interferogram import Interferogram
from src.demo.paper.haar.paper_reproduced import invert_haar
from src.demo.paper.haar.utils import generate_synthetic_spectrum, generate_interferogram, compute_wavenumbers
from src.direct_model.interferometer import FabryPerotInterferometer
from src.interface.configuration import load_config


def invert_protocols(protocols: list, wavenumbers, fp, interferogram: Interferogram):
    device = FabryPerotInterferometer(
        transmittance_coefficients=fp.transmittance,
        opds=interferogram.opds,
        phase_shift=fp.phase_shift,
        reflectance_coefficients=fp.reflectance,
        order=fp.order,
    )
    transmittance_response = device.transmittance_response(wavenumbers=wavenumbers)
    transmittance_response = transmittance_response.rescale(new_max=1., axis=None)

    spectrum_protocols = []
    for protocol in protocols:
        spectrum = protocol.option.reconstruct_spectrum(interferogram=interferogram, transmittance_response=transmittance_response)
        spectrum_protocols.append(spectrum)

    return spectrum_protocols


def load_inversion_protocols():
    config = load_config()
    db = config.database()
    protocols = [
        SimpleNamespace(option=db.inversion_protocol(inv_protocol_id=0, lambdaa=0.), label="IDCT", color="green"),
        SimpleNamespace(option=db.inversion_protocol(inv_protocol_id=1, lambdaa=0.), label="PINV", color="purple"),
        SimpleNamespace(option=db.inversion_protocol(inv_protocol_id=4, lambdaa=(10 ** 0.9)), label="LV-L1", color="orange"),
    ]
    return protocols


def load_spectrum(option: str):
    config = load_config()
    db = config.database()

    if option == "solar":
        spectrum = db.dataset_spectrum(ds_id=0)
        acq_id = 0
        spectrum = replace(spectrum, data=spectrum.data[:, acq_id:acq_id + 1])

    elif option == "specim":
        spectrum = db.dataset_spectrum(ds_id=2)
        acq_id = 13
        spectrum = replace(spectrum, data=spectrum.data[:, acq_id:acq_id + 1])

    else:
        raise ValueError(f"Option {option} is not supported.")

    return spectrum


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
    protocols = load_inversion_protocols()

    # SIMULATION

    spectrum_ref = generate_synthetic_spectrum(opd_info_obj, wn_bounds_obj, gauss_params_obj)

    # OBSERVATION

    interferogram_sim = generate_interferogram(opd_info_obj, fp_obj, spectrum_ref)
    # interferogram_sim = interferogram_sim.center(new_mean=0., axis=-2)
    interferogram_sim = interferogram_sim.rescale(new_max=1., axis=-2)
    if snr_db is not None:
        interferogram_sim = interferogram_sim.add_noise(snr_db=snr_db)

    # INVERSION

    wavenumbers = compute_wavenumbers(opd_info_obj, wn_bounds_obj)
    spectrum_haar = invert_haar(wavenumbers, fp_obj, haar_order, interferogram_sim)
    spectrum_protocols = invert_protocols(protocols, wavenumbers, fp_obj, interferogram_sim)

    # VISUALIZATION

    acq_idx = 0
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    axs_spc, axs_ifm, axs_rec = axs
    spc_ylim = [-0.1, 1.4]

    spectrum_ref = spectrum_ref.rescale(new_max=1., axis=-2)
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

    for spectrum_protocol, protocol in zip(spectrum_protocols, protocols):
        spectrum_protocol, _ = spectrum_protocol.match_stats(reference=spectrum_ref)
        spectrum_protocol.visualize(
            axs=axs_rec,
            acq_ind=acq_idx,
            label=protocol.label,
            color=protocol.color,
            linestyle="--",
            ylim=spc_ylim,
        )

    spectrum_haar, _ = spectrum_haar.match_stats(reference=spectrum_ref, axis=-2)
    spectrum_haar.visualize(
        axs=axs_rec,
        acq_ind=acq_idx,
        label="HAAR",
        color="blue",
        linestyle="",
        marker="x",
        markevery=40,
        ylim=spc_ylim,
    )

    plt.show()


if __name__ == "__main__":
    paper_test_modified()
