from src.common_utils.utils import calculate_rmse
from src.demo.experiments_paper.snr.utils import experiment_dir_convention, experiment_subdir_convention
from src.interface.configuration import load_config
from src.outputs.serialize import numpy_save_list, numpy_load_list


def main():
    config = load_config()
    db = config.database()

    is_verbose = True
    experiment_id = 3
    experiment_config = db.experiments[experiment_id]

    for ids, dataset_id in enumerate(experiment_config.dataset_ids):
        print(f"Dataset: {db.datasets[dataset_id].title.upper()}")

        for interferometer_id in experiment_config.interferometer_ids:
            print(f"\tInterferometer: {db.interferometers[interferometer_id].title.upper()}")

            noise_level_index = experiment_config.noise_level_indices[0]
            inversion_protocol_id = experiment_config.inversion_protocol_ids[0]

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

            spectra_rec, = numpy_load_list(
                filenames=["spectra_rec.npy"],
                directory=reconstruction_dir,
                subdirectory=inverter_subdir,
            )

            spectra_ref = db.dataset_spectrum(ds_id=dataset_id).data
            rmse_min = calculate_rmse(
                array=spectra_rec,
                reference=spectra_ref,
                is_match_axis=-2,
                is_match_stats=True,
                is_rescale_reference=True,
            )
            if is_verbose:
                print(f"\t\t\t\tRMSE:   {rmse_min:.3f}")

            numpy_save_list(
                filenames=["spectra_rec.npy"],
                arrays=[spectra_rec],
                directories=[metrics_dir],
                subdirectory=inverter_subdir,
            )


if __name__ == "__main__":
    main()
