from enum import Enum

import numpy as np

from src.common_utils.custom_vars import EnumInvalidOptionError
from src.common_utils.utils import calculate_rmse
from src.demo.paper.snr.utils import experiment_dir_convention, experiment_subdir_convention
from src.interface.configuration import load_config
from src.outputs.serialize import numpy_save_list, numpy_load_list


# TODO: To be automatized:
#   Directory convention


def simulate(
        experiment_id: int,
        dataset_id: int,
        interferometer_id: int,
        is_verbose: bool = False,
):
    config = load_config()
    db = config.database()

    if is_verbose:
        print(f"Dataset: {db.datasets[dataset_id].title.upper()}")
        print(f"\tInterferometer: {db.interferometers[interferometer_id].title.upper()}")

    simulation_dir = experiment_dir_convention(
        dir_type="simulation",
        experiment_id=experiment_id,
    )
    interferometer_subdir = experiment_subdir_convention(
        dataset_id=dataset_id,
        interferometer_id=interferometer_id,
    )

    spectra_ref = db.dataset_spectrum(ds_id=dataset_id)
    interferometer = db.interferometer(ifm_id=interferometer_id)
    interferograms = interferometer.acquire_interferogram(spectrum=spectra_ref)
    # TODO: interferograms.serialize()/.dump()
    numpy_save_list(
        filenames=["data.npy", "opds.npy"],
        arrays=[interferograms.data, interferograms.opds],
        directories=[simulation_dir],
        subdirectory=interferometer_subdir,
    )


def invert(
        experiment_id: int,
        dataset_id: int,
        interferometer_id: int,
        noise_level_index: int,
        inversion_protocol_id: int,
        is_verbose: bool = False,
):
    config = load_config()
    db = config.database()

    if is_verbose:
        print(f"Dataset: {db.datasets[dataset_id].title.upper()}")
        print(f"\tInterferometer: {db.interferometers[interferometer_id].title.upper()}")
        print(f"\t\tSNR: {db.noise_levels[noise_level_index]} dB")
        print(f"\t\t\tInversion Protocol: {db.inversion_protocols[inversion_protocol_id].title.upper()}")

    # TODO: Load the interferograms from a simulation
    reconstruction_dir = experiment_dir_convention(
        dir_type="reconstruction",
        experiment_id=experiment_id,
    )
    inverter_subdir = experiment_subdir_convention(
        dataset_id=dataset_id,
        interferometer_id=interferometer_id,
        noise_level_index=noise_level_index,
        inversion_protocol_id=inversion_protocol_id,
    )

    # Observation and Device
    spectra_ref = db.dataset_spectrum(ds_id=dataset_id)
    interferometer = db.interferometer(ifm_id=interferometer_id)

    # Simulate the interferograms / Acquisition
    # TODO: Interferogram.load()
    interferograms_ref = interferometer.acquire_interferogram(spectrum=spectra_ref)
    interferograms_ref = interferograms_ref.rescale(new_max=1, axis=-2)

    # Load the transfer matrix / Knowledge of the acquisition (direct) model
    transfer_matrix = interferometer.transmittance_response(wavenumbers=spectra_ref.wavenumbers)
    transfer_matrix = transfer_matrix.rescale(new_max=1, axis=None)

    # Add noise to the interferograms
    snr_db = db.noise_levels[noise_level_index]
    np.random.seed(0)
    interferograms_noisy = interferograms_ref.add_noise(snr_db=snr_db)

    # Reconstruct the spectrum / Invert the interferograms
    lambdaas = db.inversion_protocol_lambdaas(inv_protocol_id=inversion_protocol_id)
    spectra_rec_all = np.zeros(shape=(lambdaas.size, *spectra_ref.data.shape))
    for i_lmd, lambdaa in enumerate(lambdaas):
        inverter = db.inversion_protocol(inv_protocol_id=inversion_protocol_id, lambdaa=lambdaa)
        spectra_rec = inverter.reconstruct_spectrum(
            interferogram=interferograms_noisy, transmittance_response=transfer_matrix
        )
        spectra_rec_all[i_lmd] = spectra_rec.data
    rmse_lambdaas = calculate_rmse(
        array=spectra_rec_all,
        reference=spectra_ref.data,
        is_match_axis=-2,
        is_match_stats=True,
        is_rescale_reference=True,
    )
    argmin_rmse = np.argmin(rmse_lambdaas)
    spectra_rec_best = spectra_rec_all[argmin_rmse]

    # Save reconstruction
    numpy_save_list(
        filenames=["argmin_rmse.npy", "spectra_rec_best.npy"],
        arrays=[argmin_rmse, spectra_rec_best],
        directories=[reconstruction_dir],
        subdirectory=inverter_subdir,
    )


def metrics(
        experiment_id: int,
        dataset_id: int,
        interferometer_id: int,
        noise_level_index: int,
        inversion_protocol_id: int,
        is_verbose: bool = False,
):
    config = load_config()
    db = config.database()

    if is_verbose:
        print(f"Dataset: {db.datasets[dataset_id].title.upper()}")
        print(f"\tInterferometer: {db.interferometers[interferometer_id].title.upper()}")
        print(f"\t\tSNR: {db.noise_levels[noise_level_index]} dB")
        print(f"\t\t\tInversion Protocol: {db.inversion_protocols[inversion_protocol_id].title.upper()}")

    reconstruction_dir = experiment_dir_convention(
        dir_type="reconstruction",
        experiment_id=experiment_id,
    )
    metrics_dir = experiment_dir_convention(
        dir_type="metrics",
        experiment_id=experiment_id,
    )
    inverter_subdir = experiment_subdir_convention(
        dataset_id=dataset_id,
        interferometer_id=interferometer_id,
        noise_level_index=noise_level_index,
        inversion_protocol_id=inversion_protocol_id,
    )

    # Load reconstruction
    argmin, spectra_rec_best = numpy_load_list(
        filenames=["argmin_rmse.npy", "spectra_rec_best.npy"],
        directory=reconstruction_dir,
        subdirectory=inverter_subdir,
    )

    # Compute metrics
    lambdaas = db.inversion_protocol_lambdaas(inv_protocol_id=inversion_protocol_id)
    lambdaa_min = lambdaas[argmin]
    spectra_ref = db.dataset_spectrum(ds_id=dataset_id).data
    rmse_min = calculate_rmse(
        array=spectra_rec_best,
        reference=spectra_ref,
        is_match_axis=-2,
        is_match_stats=True,
        is_rescale_reference=True,
    )
    if is_verbose:
        print(f"\t\t\t\tLambda: {lambdaa_min:.3f}")
        print(f"\t\t\t\tRMSE:   {rmse_min:.3f}")

    # Save metrics
    numpy_save_list(
        filenames=["lambdaa_min", "rmse_min"],
        arrays=[lambdaa_min, rmse_min],
        directories=[metrics_dir],
        subdirectory=inverter_subdir,
    )


def visualize(
        experiment_id: int,
        dataset_id: int,
        interferometer_id: int,
        noise_level_index: int,
        inversion_protocol_id: int,
        is_verbose: bool = False,
):
    config = load_config()
    db = config.database()


class ExperimentActionEnum(str, Enum):
    SIMULATE = "simulate"
    INVERT = "invert"
    METRICS = "metrics"
    VISUALIZE = "visualize"


def action_factory(choice: ExperimentActionEnum):
    if choice == ExperimentActionEnum.SIMULATE:
        action = simulate
    elif choice == ExperimentActionEnum.INVERT:
        action = invert
    elif choice == ExperimentActionEnum.METRICS:
        action = metrics
    elif choice == ExperimentActionEnum.VISUALIZE:
        action = visualize
    else:
        raise EnumInvalidOptionError(option=choice, enum_class=ExperimentActionEnum)
    return action


def run_action(choice: ExperimentActionEnum, **inputs):
    action = action_factory(choice=choice)
    action(**inputs)


def experiment_action(experiment_id: int, action: ExperimentActionEnum):
    config = load_config()
    db = config.database()

    experiment_config = db.experiments[experiment_id]

    for ds_id in experiment_config.dataset_ids:
        for ifm_id in experiment_config.interferometer_ids:
            for nl_idx in experiment_config.noise_level_indices:
                for ip_id in experiment_config.inversion_protocol_ids:
                    run_action(
                        choice=action,
                        experiment_id=experiment_id,
                        dataset_id=ds_id,
                        interferometer_id=ifm_id,
                    )


def main():
    experiment_id = 0
    action = ExperimentActionEnum.SIMULATE

    experiment_action(experiment_id=experiment_id, action=action)


if __name__ == "__main__":
    main()
