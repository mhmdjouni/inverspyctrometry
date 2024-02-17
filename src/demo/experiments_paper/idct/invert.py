import numpy as np

from src.demo.experiments_paper.snr.utils import experiment_dir_convention, experiment_subdir_convention
from src.interface.configuration import load_config
from src.outputs.serialize import numpy_save_list


def main():
    config = load_config()
    db = config.database()

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
            inverter_subdir = experiment_subdir_convention(
                dataset_id=dataset_id,
                interferometer_id=interferometer_id,
                noise_level_index=noise_level_index,
                inversion_protocol_id=inversion_protocol_id,
            )

            spectra_ref = db.dataset_spectrum(ds_id=dataset_id)
            interferometer = db.interferometer(ifm_id=interferometer_id)

            interferograms_ref = interferometer.acquire_interferogram(spectrum=spectra_ref)
            interferograms_ref = interferograms_ref.rescale(new_max=1, axis=-2)
            interferograms_ref = interferograms_ref.center(new_mean=0, axis=-2)

            transfer_matrix = interferometer.transmittance_response(wavenumbers=spectra_ref.wavenumbers)
            transfer_matrix = transfer_matrix.rescale(new_max=1, axis=None)

            snr_db = db.noise_levels[noise_level_index]
            np.random.seed(0)
            interferograms_noisy = interferograms_ref.add_noise(snr_db=snr_db)

            lambdaa = db.inversion_protocol_lambdaas(inv_protocol_id=inversion_protocol_id)[0]
            inverter = db.inversion_protocol(inv_protocol_id=inversion_protocol_id, lambdaa=lambdaa)
            spectra_rec = inverter.reconstruct_spectrum(
                interferogram=interferograms_noisy, transmittance_response=transfer_matrix
            )

            numpy_save_list(
                filenames=["spectra_rec.npy"],
                arrays=[spectra_rec.data],
                directories=[reconstruction_dir],
                subdirectory=inverter_subdir,
            )


if __name__ == "__main__":
    main()
