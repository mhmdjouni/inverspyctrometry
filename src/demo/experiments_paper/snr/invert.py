"""
Minimal working template for simulated inversion, without comments and extra functionality (saving arrays, etc.)
"""
from pprint import pprint

import numpy as np

from src.common_utils.utils import calculate_rmse
from src.direct_model.interferometer import simulate_interferogram
from src.interface.configuration import load_config
from src.outputs.visualization import numpy_save_list


def main():
    config = load_config()
    db = config.database()

    reports_folder = config.directory_paths.reports

    experiment_id = 0
    experiment_config = db.experiments[experiment_id]
    pprint(dict(experiment_config))
    experiment_dir = reports_folder / f"experiment_{experiment_id}" / "reconstruction"

    for ds_id in experiment_config.dataset_ids:
        spectra_ref = db.dataset_spectrum(ds_id=ds_id)
        print(f"Dataset: {db.datasets[ds_id].title.upper()}")
        dataset_subdir = f"invert_{db.datasets[ds_id].title}"

        for ifm_id in experiment_config.interferometer_ids:
            interferometer = db.interferometer(ifm_id=ifm_id)
            print(f"\tInterferometer: {db.interferometers[ifm_id].title.upper()}")
            interferometer_subdir = f"{dataset_subdir}/{db.interferometers[ifm_id].title}"

            transfer_matrix = interferometer.transmittance_response(wavenumbers=spectra_ref.wavenumbers)
            interferograms_ref = simulate_interferogram(transmittance_response=transfer_matrix, spectrum=spectra_ref)

            interferograms_ref = interferograms_ref.rescale(new_max=1, axis=-2)
            transfer_matrix = transfer_matrix.rescale(new_max=1, axis=None)

            for nl_idx in experiment_config.noise_level_indices:
                snr_db = db.noise_levels[nl_idx]
                print(f"\t\tSNR: {snr_db} dB")
                noise_level_subdir = f"{interferometer_subdir}/{snr_db}_db"

                np.random.seed(0)
                interferograms_noisy = interferograms_ref.add_noise(snr_db=snr_db)

                for ip_id in experiment_config.inversion_protocol_ids:
                    lambdaas = db.inversion_protocol_lambdaas(inv_protocol_id=ip_id)
                    print(f"\t\tInversion Protocol: {db.inversion_protocols[ip_id].title.upper()}")
                    inverter_subdir = f"{noise_level_subdir}/{db.inversion_protocols[ip_id].title}"

                    spectra_rec_all = np.zeros(shape=(lambdaas.size, *spectra_ref.data.shape))

                    for i_lmd, lambdaa in enumerate(lambdaas):
                        inverter = db.inversion_protocol(inv_protocol_id=ip_id, lambdaa=lambdaa)
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

                    # Save the best reconstruction, per inv_protocol per dataset
                    numpy_save_list(
                        filenames=["spectra_rec_best", "argmin_rmse"],
                        arrays=[spectra_rec_best, argmin_rmse],
                        directories_list=[experiment_dir],
                        subdirectory=inverter_subdir,
                    )

                # Save the noisy interferograms
                numpy_save_list(
                    filenames=["interferograms_noisy"],
                    arrays=[interferograms_noisy.data],
                    directories_list=[experiment_dir],
                    subdirectory=noise_level_subdir,
                )


if __name__ == "__main__":
    main()
