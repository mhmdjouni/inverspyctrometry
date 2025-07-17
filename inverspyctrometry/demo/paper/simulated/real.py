from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from matplotlib import pyplot as plt

from inverspyctrometry.common_utils.interferogram import Interferogram
from inverspyctrometry.common_utils.utils import calculate_rmse
from inverspyctrometry.demo.paper.monochromatic.utils import calculate_rmcw
from inverspyctrometry.demo.paper.simulated.simulated import compose_dir, compose_subdir, reconstruction_save_numpy, \
    reconstruction_load_numpy
from inverspyctrometry.demo.paper.simulated.utils import invert_haar, load_spectrum, invert_protocols, Protocol
from inverspyctrometry.interface.configuration import load_config
from inverspyctrometry.outputs.serialize import numpy_save_list


@dataclass(frozen=True)
class Extrapolation:
    case: int
    description: str
    opds_resampler: str
    extrap_kind: str
    extrap_fill: any
    transmat_extrap: str


def load_interferogram(option: str) -> tuple[Interferogram, np.ndarray]:
    if option == "mc451":
        dataset_id = 3
        # acq_id = 200
    elif option == "mc651":
        dataset_id = 4
        # acq_id = 300
    elif option == "cc_green":
        dataset_id = 5
        # acq_id = 0
    else:
        raise ValueError(f"Option {option} is not supported.")

    db = load_config().database()
    interferogram = db.dataset_interferogram(ds_id=dataset_id)
    central_wavenumbers = db.dataset_central_wavenumbers(dataset_id=dataset_id)
    # interferogram = replace(interferogram, data=interferogram.data[:, acq_id:acq_id + 1])
    return interferogram, central_wavenumbers


def get_interferogram(ifm_type: str, extrap: Extrapolation):
    """
    Load the Interferogram dataset of interest and the central wavenumbers
    Sort the OPDs
    Extrapolate the OPDs starting from the Zero-OPD
    """
    interferogram, central_wavenumbers = load_interferogram(option=ifm_type)
    # interferogram = interferogram.sort_opds()
    if extrap is not None:
        interferogram = interferogram.extrapolate(
            support_resampler=extrap.opds_resampler,
            kind=extrap.extrap_kind,
            fill_value=extrap.extrap_fill,
        )
    return interferogram


def invert_haar_real(wavenumbers_central, characterization_id, haar_order, interferogram_sim):
    db = load_config().database()
    characterization = db.characterization(characterization_id=characterization_id)
    characterization = characterization.sort_opds()
    transmittance = characterization.transmittance(wavenumbers=wavenumbers_central).mean(keepdims=True)
    reflectance = characterization.reflectance(wavenumbers=wavenumbers_central).mean(keepdims=True)
    fp_obj = SimpleNamespace(
        transmittance=transmittance,
        reflectance=reflectance,
    )
    spectrum, execution_time = invert_haar(wavenumbers_central, fp_obj, haar_order, interferogram_sim)
    return spectrum, execution_time


def invert_protocols_real(protocols, wavenumbers, characterization_id, interferogram, spectrum_ref,
                          extrap: Extrapolation):
    db = load_config().database()
    characterization = db.characterization(characterization_id=characterization_id)
    # characterization = characterization.sort_opds()
    if extrap is not None:
        characterization = characterization.extrapolate_opds(support_resampler=extrap.opds_resampler)
    fp_obj = SimpleNamespace(
        transmittance=characterization.transmittance_coefficients,
        phase_shift=characterization.phase_shift,
        reflectance=characterization.reflectance_coefficients,
        order=characterization.order,
    )
    spectrum_protocols, argmin_rmses, execution_time_protocols, rmse_lambdaas_protocols, cost_progress_protocols = invert_protocols(
        protocols,
        wavenumbers,
        fp_obj,
        interferogram,
        spectrum_ref=spectrum_ref,
    )
    return spectrum_protocols, argmin_rmses, execution_time_protocols, rmse_lambdaas_protocols, cost_progress_protocols


@dataclass
class Options:
    ifm_types: list[str]  # ["mc451", "mc651", "cc_green"]
    device_name: str
    name: str


def main():
    # Options 0: Test with low, medium, high, and variable reflectivity
    # Options 1: Test with regular vs irregular sampling in the OPDs
    # Options 2: Test with [20, 15] dB of noise
    options_list = [
        Options(
            name="real",
            device_name="imspoc_uv_2",
            ifm_types=["mc451", "mc651"],
        ),
    ]

    options = options_list[0]
    for ifm_type in options.ifm_types:
        experiment_run(
            ifm_type=ifm_type,
            device_name=options.device_name,
            experiment_name=options.name,
        )


def metrics_real_save_numpy(
        lambdaa_min,
        rmse_full,
        rmse_diagonal,
        rmcw,
        execution_time_min,
        cost_progress_min,
        directories,
        subdirectory,
):
    numpy_save_list(
        filenames=["lambdaa_min.npy", "rmse_full.npy", "rmse_diagonal.npy", "rmcw.npy", "execution_time_min.npy",
                   "cost_progress_min.npy"],
        arrays=[lambdaa_min, rmse_full, rmse_diagonal, rmcw, execution_time_min, cost_progress_min],
        directories=directories,
        subdirectory=subdirectory,
    )


def experiment_run(
        experiment_name: str,
        device_name: str,
        ifm_type: str,
):
    # OPTIONS

    ifm_type = ifm_type
    extrap = Extrapolation(
        case=3,
        description="Concatenate lowest OPDs but extrapolate the interferogram values using fourier series",
        opds_resampler="concatenate_missing",
        extrap_kind="linear",
        extrap_fill="fourier",
        transmat_extrap="model",
    )
    extrap = None
    char_id = 0
    haar_order = 20
    protocols = [
        # Protocol(id=0, label="IDCT", color="green"),
        # Protocol(id=19, label="HAAR", color="red"),
        Protocol(id=5, label="TSVD", color="pink"),
        Protocol(id=6, label="RR", color="orange"),
        # Protocol(id=10, label="LV-L1", color="purple"),
    ]

    # OBSERVATION (SIMULATION)

    print("\n\nOBSERVATION")
    interferogram_extrap = get_interferogram(ifm_type, extrap)
    interferogram_extrap = interferogram_extrap.rescale(new_max=1., axis=-2)
    fig, axs = plt.subplots()
    interferogram_extrap.visualize_matrix(fig, axs)
    plt.show()

    directories = [
        compose_dir(report_type="extrapolation", experiment_name=experiment_name)
    ]
    subdirectory = compose_subdir(
        dataset_name=ifm_type,
        device_name=device_name,
        noise_level=None,
        protocol_name=None,
    )
    interferogram_extrap.save_numpy(
        directories=directories,
        subdirectory=subdirectory,
    )

    # REFERENCE SPECTRUM

    print("\n\nREFERENCE SPECTRUM")
    spectrum_ref = load_spectrum(option=ifm_type)
    spectrum_ref = spectrum_ref.rescale(new_max=1., axis=-2)

    # INVERSION

    print("\n\nINVERSION")
    directory = compose_dir(report_type="extrapolation", experiment_name=experiment_name)
    subdirectory = compose_subdir(
        dataset_name=ifm_type,
        device_name=device_name,
        noise_level=None,
        protocol_name=None,
    )
    interferogram_extrap = Interferogram.load_numpy(
        directory=directory,
        subdirectory=subdirectory,
    )

    wavenumbers = spectrum_ref.wavenumbers
    spectrum_haar, execution_time_haar = invert_haar_real(wavenumbers, char_id, haar_order, interferogram_extrap)
    spectrum_protocols, argmin_rmses, execution_time_protocols, rmse_lambdaas_protocols, cost_progress_protocols = invert_protocols_real(
        protocols,
        wavenumbers,
        char_id,
        interferogram_extrap,
        spectrum_ref,
        extrap,
    )
    spectrum_protocols.insert(1, spectrum_haar)
    argmin_rmses.insert(1, 0)
    execution_time_protocols.insert(1, execution_time_haar)
    rmse_lambdaas_protocols.insert(1, np.array([0.]))
    cost_progress_protocols.insert(1, np.array([0.]))

    directories = [
        compose_dir(report_type="reconstruction", experiment_name=experiment_name)
    ]
    for spectrum_protocol, protocol, argmin_rmse, execution_time_protocol, rmse_lambdaas_protocol, cost_progress_protocol in zip(
            spectrum_protocols,
            protocols,
            argmin_rmses,
            execution_time_protocols,
            rmse_lambdaas_protocols,
            cost_progress_protocols,
    ):
        subdirectory = compose_subdir(
            dataset_name=ifm_type,
            device_name=device_name,
            noise_level=None,
            protocol_name=protocol.label.lower(),
        )
        reconstruction_save_numpy(
            spectra_rec_best=spectrum_protocol,
            argmin_rmse=argmin_rmse,
            execution_time_best=execution_time_protocol,
            rmse_lambdaas_all=rmse_lambdaas_protocol,
            cost_progress_best=cost_progress_protocol,
            directories=directories,
            subdirectory=subdirectory,
        )

    # METRICS

    print("\n\nMETRICS")
    print(
        ""
        "Lambdaa"
        "DIAG"
        "FULL"
        "No.MCW"
    )
    directory = compose_dir(report_type="reconstruction", experiment_name=experiment_name)
    directories = [
        compose_dir(report_type="metrics", experiment_name=experiment_name)
    ]
    for protocol in protocols:
        subdirectory = compose_subdir(
            dataset_name=ifm_type,
            device_name=device_name,
            noise_level=None,
            protocol_name=protocol.label.lower(),
        )
        spectra_rec_best, argmin_rmse, execution_time_best, rmse_lambdaas_all, cost_progress_best = reconstruction_load_numpy(
            directory=directory,
            subdirectory=subdirectory,
        )

        lambdaas = load_config().database().inversion_protocol_lambdaas(inv_protocol_id=protocol.id)
        lambdaa_min = lambdaas[argmin_rmse]
        spectra_rec_best, _ = spectra_rec_best.match_stats(reference=spectrum_ref)
        rmse_full = calculate_rmse(
            array=spectra_rec_best.data,
            reference=spectrum_ref.data,
        )
        rmse_diagonal = calculate_rmse(
            array=np.diag(spectra_rec_best.data),
            reference=np.diag(spectrum_ref.data),
        )
        rmcw = calculate_rmcw(monochromatic_array=spectra_rec_best.data[None, ...])[0]
        print(
            f"\t{protocol.label + ':':6}, "
            f"{lambdaa_min:10.4f}, "
            f"{rmse_diagonal:10.4f}, "
            f"{rmse_full:10.4f}, "
            f"{rmcw:10.4f}"
        )

        subdirectory = compose_subdir(
            dataset_name=ifm_type,
            device_name=device_name,
            noise_level=None,
            protocol_name=protocol.label.lower(),
        )
        metrics_real_save_numpy(
            lambdaa_min=lambdaa_min,
            rmse_full=rmse_full,
            rmse_diagonal=rmse_diagonal,
            rmcw=rmcw,
            execution_time_min=execution_time_best,
            cost_progress_min=cost_progress_best,
            directories=directories,
            subdirectory=subdirectory,
        )

    # VISUALIZATION

    print("\n\nVISUALIZATION")
    acq_idx = 200
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

    interferogram_extrap.visualize(
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
