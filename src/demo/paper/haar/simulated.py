from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum
from src.common_utils.utils import calculate_rmse
from src.demo.paper.haar.paper_reproduced import invert_haar
from src.demo.paper.haar.utils import generate_synthetic_spectrum, generate_interferogram, compute_wavenumbers, \
    oversample_wavenumbers, oversample_spectrum
from src.direct_model.interferometer import FabryPerotInterferometer
from src.interface.configuration import load_config


def invert_protocols(protocols: list, wavenumbers, fp, interferogram: Interferogram, spectrum_ref: Spectrum):
    device = FabryPerotInterferometer(
        transmittance_coefficients=fp.transmittance,
        opds=interferogram.opds,
        phase_shift=fp.phase_shift,
        reflectance_coefficients=fp.reflectance,
        order=fp.order,
    )
    transmittance_response = device.transmittance_response(wavenumbers=wavenumbers)
    transmittance_response = transmittance_response.rescale(new_max=1., axis=None)

    db = load_config().database()
    spectrum_protocols = []
    for protocol in protocols:
        lambdaas = db.inversion_protocol_lambdaas(inv_protocol_id=protocol.id)
        spectrum_rec_all = np.zeros(shape=(lambdaas.size, *spectrum_ref.data.shape))
        for i_lmd, lambdaa in enumerate(lambdaas):
            inverter = db.inversion_protocol(inv_protocol_id=protocol.id, lambdaa=lambdaa)
            spectra_rec = inverter.reconstruct_spectrum(
                interferogram=interferogram, transmittance_response=transmittance_response
            )
            spectrum_rec_all[i_lmd] = spectra_rec.data

        rmse_lambdaas = calculate_rmse(
            array=spectrum_rec_all,
            reference=spectrum_ref.data,
            is_match_axis=-2,
            is_match_stats=True,
            is_rescale_reference=True,
        )
        argmin_rmse = np.argmin(rmse_lambdaas)
        print(f"{protocol.label}: lmd = {lambdaas[argmin_rmse]:.4f} at idx = {argmin_rmse:.0f}")
        spectrum_rec_best = replace(spectrum_ref, data=spectrum_rec_all[argmin_rmse])
        spectrum_protocols.append(spectrum_rec_best)

    return spectrum_protocols


def load_spectrum(option: str):
    config = load_config()
    db = config.database()
    if option == "solar":
        spectrum = db.dataset_spectrum(ds_id=0)
        acq_id = 0
    elif option == "specim":
        spectrum = db.dataset_spectrum(ds_id=2)
        acq_id = 13
    else:
        raise ValueError(f"Option {option} is not supported.")
    spectrum = replace(spectrum, data=spectrum.data[:, acq_id:acq_id + 1])
    return spectrum


def load_opd_info(device: str):
    if device == "paper":
        opd_info = SimpleNamespace(
            step=np.round(100 * 1e-7, decimals=5),  # 100 nm => cm
            num=2048,
            unit="cm",
        )
    elif device == "imspoc_uv_2":
        opd_info = SimpleNamespace(  # imspoc uv 2
            step=np.round(175 * 1e-3, decimals=3),  # 175 nm => um
            num=319,
            unit="um",
        )
    else:
        raise ValueError(f"{device}")
    return opd_info


def get_spectrum(spc_type: str):
    if spc_type == "paper":
        wn_bounds_obj = SimpleNamespace(
            start=0,  # cm-1
            stop=20000.1,  # cm-1
            unit="1/cm",
        )  # nm
        gauss_params_obj = SimpleNamespace(
            coeffs=np.array([1., 0.9, 0.75]),
            means=np.array([2000, 4250, 6500]),  # cm-1
            stds=np.array([300, 1125, 400]),  # cm-1
        )
        opd_info_obj = load_opd_info(device="paper")
        spectrum = generate_synthetic_spectrum(gauss_params_obj, opd_info_obj, wn_bounds_obj)
    elif spc_type in ["solar", "specim"]:
        opd_info_obj = load_opd_info(device="imspoc_uv_2")
        spectrum = load_spectrum(spc_type)
    else:
        raise ValueError(f"{spc_type}")
    power_2 = int(np.ceil(np.log2(spectrum.wavenumbers.size)))
    size_new = 2 ** power_2
    wavenumbers_new = np.linspace(
        start=spectrum.wavenumbers[0],
        stop=spectrum.wavenumbers[-1],
        num=size_new,
        endpoint=True,
    )
    spectrum = spectrum.interpolate(wavenumbers=wavenumbers_new, kind="slinear")
    return spectrum, opd_info_obj


def main():
    # OPTIONS

    options = SimpleNamespace(
        spc_type="specim",  # "paper", "solar", "specim"
    )
    fp_obj = SimpleNamespace(
        transmittance=np.array([1.]),
        phase_shift=np.array([0.]),
        reflectance=np.array([0.2]),
        order=0,
    )
    snr_db = None
    is_oversample_first = True
    haar_order = 20
    wn_num_factor = 10
    protocols = [
        SimpleNamespace(id=0, label="IDCT", color="green"),
        SimpleNamespace(id=1, label="PINV", color="black"),
        SimpleNamespace(id=2, label="TSVD", color="C3"),
        SimpleNamespace(id=3, label="RR", color="orange"),
        SimpleNamespace(id=4, label="LV-L1", color="purple"),
    ]

    # SPECTRUM

    print("\n\nSPECTRUM")
    spectrum_ref, opd_info_obj = get_spectrum(options.spc_type)
    if is_oversample_first:
        spectrum_ref = oversample_spectrum(spectrum_ref, opd_info_obj, wn_num_factor)
    spectrum_ref = spectrum_ref.rescale(new_max=1., axis=-2)

    # OBSERVATION (SIMULATION)

    print("\n\nOBSERVATION")
    interferogram_sim = generate_interferogram(opd_info_obj, fp_obj, spectrum_ref, wn_num_factor)
    interferogram_sim = interferogram_sim.rescale(new_max=1., axis=-2)
    if snr_db is not None:
        protocols.pop(1)
        np.random.seed(0)
        interferogram_sim = interferogram_sim.add_noise(snr_db=snr_db)

    # INVERSION

    print("\n\nINVERSION")
    wavenumbers = spectrum_ref.wavenumbers
    spectrum_haar = invert_haar(wavenumbers, fp_obj, haar_order, interferogram_sim)
    spectrum_protocols = invert_protocols(protocols, wavenumbers, fp_obj, interferogram_sim, spectrum_ref=spectrum_ref)

    # METRICS

    print("\n\nMETRICS")
    rmse = calculate_rmse(
        array=spectrum_haar.data,
        reference=spectrum_ref.data,
        is_match_axis=-2,
        is_match_stats=True,
        is_rescale_reference=True,
    )
    print(f"\t{'HAAR:':6} RMSE = {rmse:.4f}")
    for spectrum_protocol, protocol in zip(spectrum_protocols, protocols):
        rmse = calculate_rmse(
            array=spectrum_protocol.data,
            reference=spectrum_ref.data,
            is_match_axis=-2,
            is_match_stats=True,
            is_rescale_reference=True,
        )
        print(f"\t{protocol.label + ':':6} RMSE = {rmse:.4f}")

    # VISUALIZATION

    print("\n\nVISUALIZATION")
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
        linestyle=":",
        marker="x",
        markevery=40,
        ylim=spc_ylim,
    )

    plt.show()


if __name__ == "__main__":
    main()
