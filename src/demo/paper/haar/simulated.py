from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from src.common_utils.custom_vars import Opd
from src.common_utils.interferogram import Interferogram
from src.common_utils.light_wave import Spectrum
from src.common_utils.utils import calculate_rmse, polyval_rows
from src.demo.paper.haar.utils import generate_synthetic_spectrum, generate_interferogram, oversample_spectrum, \
    invert_haar, \
    load_spectrum, invert_protocols, OPDummy, Protocol
from src.interface.configuration import load_config
from src.outputs.serialize import numpy_save_list, numpy_load_list


def load_opd_info(dataset: str):
    if dataset == "paper":
        opd_info = SimpleNamespace(
            step=np.round(100 * 1e-7, decimals=5),  # 100 nm => cm
            num=2048,
            unit="cm",
        )
    elif dataset in ["solar", "specim", "shine"]:
        opd_info = SimpleNamespace(  # imspoc uv 2
            step=np.round(175 * 1e-3, decimals=3),  # 175 nm => um
            num=319,
            unit="um",
        )
    else:
        raise ValueError(f"{dataset}")
    return opd_info


def load_opds(opds_sampling: str, dataset: str) -> OPDummy:
    if opds_sampling == "regular":
        opd_info = load_opd_info(dataset=dataset)
        opds = OPDummy.from_opd_info(step=opd_info.step, num=opd_info.num, unit=opd_info.unit)
    elif opds_sampling == "irregular":
        opds_arr = load_real_opds()
        opds = OPDummy(data=opds_arr, unit="um")
    else:
        raise ValueError(f"{dataset}")
    return opds


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
        opd_info_obj = load_opd_info(dataset="paper")
        spectrum = generate_synthetic_spectrum(gauss_params_obj, opd_info_obj, wn_bounds_obj)
    elif spc_type in ["solar", "specim", "shine"]:
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
    return spectrum


@dataclass(frozen=True)
class Options:
    experiment_name: str
    fp_tr: list[tuple[str, np.ndarray, np.ndarray]]
    noise: list[Optional[float]]
    opds_sampling: str
    spc_types: list[str]
    protocols: list[Protocol]


def load_variable_reflectivity() -> tuple[
    str, np.ndarray[tuple[int], np.dtype[np.float_]], np.ndarray[tuple[int], np.dtype[np.float_]]
]:
    characterization = load_config().database().characterization(char_id=0)
    ifm_idx = 30
    reflectivity_coeffs = characterization.reflectance_coefficients[ifm_idx:ifm_idx+1].mean(axis=-2, keepdims=True)
    reflectivity_coeffs[0][0] += 0.3
    transmissivity_coeffs = - reflectivity_coeffs
    transmissivity_coeffs[0][0] += 1

    wavenumbers = np.linspace(0.666, 2.9, int(1e4))
    transmissivity = polyval_rows(coefficients=transmissivity_coeffs, interval=wavenumbers)
    reflectivity = polyval_rows(coefficients=reflectivity_coeffs, interval=wavenumbers)
    plt.plot(wavenumbers, transmissivity[0])
    plt.plot(wavenumbers, reflectivity[0])
    plt.grid()
    # plt.ylim([-0.2, 1.2])
    plt.show()

    return "fp_0_var_r", transmissivity_coeffs, reflectivity_coeffs


def load_real_opds() -> np.ndarray[tuple[Opd], np.dtype[np.float_]]:
    opds = load_config().database().characterization(char_id=0).opds

    opds = np.sort(opds)
    opd_mean_step = np.mean(np.diff(opds))
    lowest_missing_opds = np.arange(start=0., stop=opds.min(), step=opd_mean_step)
    opds = np.concatenate((lowest_missing_opds, opds))

    return opds


def compose_save_dir(
        report_type: str,
        experiment_name: str,
        save_dir_init: str = None,
):
    if save_dir_init is None:
        save_dir = load_config().directory_paths.reports
    else:
        save_dir = save_dir_init

    if report_type in ["oversampling", "simulation", "reconstruction", "metrics", "figures"]:
        return save_dir / experiment_name / report_type
    elif report_type == "paper_figures":
        return save_dir / "figures" / experiment_name
    else:
        raise ValueError(f"Report type '{report_type}' is not supported.")


def compose_subdir(
        dataset_name: str,
        device_name: str,
        noise_level: Optional[float],
        protocol_name: Optional[str],
        subdir_init: Optional[str] = None,
):
    if subdir_init is None:
        subdir = "."
    else:
        subdir = subdir_init
    subdir += f"/{dataset_name}"
    subdir += f"/{device_name}"
    if noise_level is not None:
        subdir += f"/{int(noise_level)}_db"
    if protocol_name is not None:
        subdir += f"/{protocol_name}"
    return subdir


def main():
    # Options 0: Test with low, medium, high, and variable reflectivity
    # Options 1: Test with regular vs irregular sampling in the OPDs
    # Options 2: Test with [20, 15, 10] dB of noise
    options_list = [
        Options(
            experiment_name="simulated/reflectivity_levels",
            fp_tr=[
                ("fp_0_low_r", np.array([[1.]]), np.array([[0.2]])),
                ("fp_0_med_r", np.array([[1.]]), np.array([[0.4]])),
                ("fp_0_hig_r", np.array([[1.]]), np.array([[0.7]])),
                load_variable_reflectivity(),
            ],
            noise=[None],
            opds_sampling="regular",
            spc_types=["solar", "specim", "shine"],
            protocols=[
                Protocol(id=0, label="IDCT", color="green"),
                Protocol(id=1, label="PINV", color="black"),
            ],
        ),
        Options(
            experiment_name="simulated/irregular_sampling",
            fp_tr=[
                ("fp_0_low_r", np.array([[1.]]), np.array([[0.2]])),
            ],
            noise=[None],
            opds_sampling="irregular",
            spc_types=["solar", "specim", "shine"],
            protocols=[
                Protocol(id=0, label="IDCT", color="green"),
                Protocol(id=1, label="PINV", color="purple"),
            ],
        ),
        Options(
            experiment_name="simulated/noise_levels",
            fp_tr=[
                ("fp_0_low_r", np.array([[1.]]), np.array([[0.2]])),
            ],
            noise=[
                20.,
                15.,
                10.,
            ],
            opds_sampling="regular",
            spc_types=["solar", "specim", "shine"],
            protocols=[
                Protocol(id=0, label="IDCT", color="green"),
                Protocol(id=2, label="TSVD", color="C3"),
                Protocol(id=3, label="RR", color="orange"),
                Protocol(id=4, label="LV-L1", color="black"),
            ],
        ),
    ]

    options = options_list[0]
    for noise in options.noise:
        for device_name, transmissivity, reflectivity in options.fp_tr:
            for spc_type in options.spc_types:
                experiment(
                    name=options.experiment_name,
                    device_name=device_name,
                    transmissivity=transmissivity,
                    reflectivity=reflectivity,
                    noise=noise,
                    opds_sampling=options.opds_sampling,
                    spc_type=spc_type,
                    protocols=options.protocols,
                )


def reconstruction_save_numpy(
        spectra_rec_best: Spectrum,
        argmin_rmse: np.ndarray,
        directories: list[Path],
        subdirectory: str,
):
    spectra_rec_best.save_numpy(
        directories=directories,
        subdirectory=subdirectory,
    )
    numpy_save_list(
        filenames=["argmin_rmse.npy"],
        arrays=[argmin_rmse],
        directories=directories,
        subdirectory=subdirectory,
    )


def reconstruction_load_numpy(directory, subdirectory) -> tuple[Spectrum, np.ndarray]:
    spectra_rec_best = Spectrum.load_numpy(directory=directory, subdirectory=subdirectory)
    argmin_rmse, = numpy_load_list(
        filenames=["argmin_rmse.npy"],
        directory=directory,
        subdirectory=subdirectory,
    )
    return spectra_rec_best, argmin_rmse


def metrics_save_numpy(lambdaa_min, rmse_min, directories, subdirectory):
    numpy_save_list(
        filenames=["lambdaa_min.npy", "rmse_min.npy"],
        arrays=[lambdaa_min, rmse_min],
        directories=directories,
        subdirectory=subdirectory,
    )


def experiment(
        name: str,
        device_name: str,
        transmissivity: np.ndarray,
        reflectivity: np.ndarray,
        noise: float,
        opds_sampling: str,
        spc_type: str,
        protocols: list[Protocol],
):
    # OPTIONS

    options = SimpleNamespace(
        spc_type=spc_type,  # "paper", "solar", "specim"
        opds_sampling=opds_sampling,
    )
    fp_obj = SimpleNamespace(
        transmittance=transmissivity,
        phase_shift=np.array([0.]),
        reflectance=reflectivity,
        order=0,
    )
    snr_db = noise
    is_oversample_first = True
    wn_num_factor = 10
    haar_order = 20
    protocols = protocols

    # SPECTRUM

    print("\n\nSPECTRUM")
    spectrum_ref = get_spectrum(options.spc_type)

    # OVERSAMPLING WAVENUMBERS

    opds = load_opds(opds_sampling=options.opds_sampling, dataset=options.spc_type)
    if is_oversample_first:
        spectrum_ref = oversample_spectrum(spectrum_ref, opds, wn_num_factor)
    spectrum_ref = spectrum_ref.rescale(new_max=1., axis=-2)

    directories = [
        compose_save_dir(report_type="oversampling", experiment_name=name)
    ]
    subdirectory = compose_subdir(
        dataset_name=options.spc_type,
        device_name=device_name,
        noise_level=None,
        protocol_name=None,
    )
    spectrum_ref.save_numpy(
        directories=directories,
        subdirectory=subdirectory,
    )

    # OBSERVATION (SIMULATION)

    print("\n\nOBSERVATION")
    directory = compose_save_dir(report_type="oversampling", experiment_name=name)
    subdirectory = compose_subdir(
        dataset_name=options.spc_type,
        device_name=device_name,
        noise_level=None,
        protocol_name=None,
    )
    spectrum_ref = Spectrum.load_numpy(
        directory=directory,
        subdirectory=subdirectory,
    )
    fig, axs = plt.subplots(1, 1)
    spectrum_ref.visualize(axs=axs, acq_ind=0)
    plt.show()

    interferogram_sim = generate_interferogram(opds, fp_obj, spectrum_ref, wn_num_factor)
    interferogram_sim = interferogram_sim.rescale(new_max=1., axis=-2)

    directories = [
        compose_save_dir(report_type="simulation", experiment_name=name)
    ]
    subdirectory = compose_subdir(
        dataset_name=options.spc_type,
        device_name=device_name,
        noise_level=None,
        protocol_name=None,
    )
    interferogram_sim.save_numpy(
        directories=directories,
        subdirectory=subdirectory,
    )

    # INVERSION

    print("\n\nINVERSION")
    directory = compose_save_dir(report_type="simulation", experiment_name=name)
    subdirectory = compose_subdir(
        dataset_name=options.spc_type,
        device_name=device_name,
        noise_level=None,
        protocol_name=None,
    )
    interferogram_sim = Interferogram.load_numpy(
        directory=directory,
        subdirectory=subdirectory,
    )
    fig, axs = plt.subplots(1, 1)
    interferogram_sim.visualize(axs=axs, acq_ind=0)
    plt.show()

    if snr_db is not None:
        np.random.seed(0)
        interferogram_sim = interferogram_sim.add_noise(snr_db=snr_db)
    wavenumbers = spectrum_ref.wavenumbers
    spectrum_haar = invert_haar(wavenumbers, fp_obj, haar_order, interferogram_sim)
    spectrum_protocols, argmin_rmses = invert_protocols(protocols, wavenumbers, fp_obj, interferogram_sim, spectrum_ref=spectrum_ref)
    spectrum_protocols.insert(1, spectrum_haar)
    argmin_rmses.insert(1, 0)
    protocols.insert(1, Protocol(id=19, label="HAAR", color="blue"))

    directories = [
        compose_save_dir(report_type="reconstruction", experiment_name=name)
    ]
    for spectrum_protocol, protocol, argmin_rmse in zip(spectrum_protocols, protocols, argmin_rmses):
        subdirectory = compose_subdir(
            dataset_name=options.spc_type,
            device_name=device_name,
            noise_level=snr_db,
            protocol_name=protocol.label.lower(),
        )
        reconstruction_save_numpy(
            spectra_rec_best=spectrum_protocol,
            argmin_rmse=argmin_rmse,
            directories=directories,
            subdirectory=subdirectory,
        )

    # METRICS

    print("\n\nMETRICS")
    directory = compose_save_dir(report_type="reconstruction", experiment_name=name)
    directories = [
        compose_save_dir(report_type="metrics", experiment_name=name)
    ]
    for protocol in protocols:
        subdirectory = compose_subdir(
            dataset_name=options.spc_type,
            device_name=device_name,
            noise_level=snr_db,
            protocol_name=protocol.label.lower(),
        )
        spectra_rec_best, argmin_rmse = reconstruction_load_numpy(
            directory=directory,
            subdirectory=subdirectory,
        )

        lambdaas = load_config().database().inversion_protocol_lambdaas(inv_protocol_id=protocol.id)
        lambdaa_min = lambdaas[argmin_rmse]
        rmse_min = calculate_rmse(
            array=spectra_rec_best.data,
            reference=spectrum_ref.data,
            is_match_axis=-2,
            is_match_stats=True,
            is_rescale_reference=True,
        )
        print(f"\t{protocol.label + ':':6} RMSE = {rmse_min:.4f}, Lambda: {lambdaa_min:.4f}")

        subdirectory = compose_subdir(
            dataset_name=options.spc_type,
            device_name=device_name,
            noise_level=snr_db,
            protocol_name=protocol.label.lower(),
        )
        metrics_save_numpy(
            lambdaa_min=lambdaa_min,
            rmse_min=rmse_min,
            directories=directories,
            subdirectory=subdirectory,
        )

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
