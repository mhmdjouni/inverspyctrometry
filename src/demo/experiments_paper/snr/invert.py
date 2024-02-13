"""
Exhaustive working template for simulated inversion, with comments and extra functionality (saving arrays, etc.)
"""

from pprint import pprint

import numpy as np

from src.common_utils.utils import calculate_rmse
from src.direct_model.interferometer import simulate_interferogram
from src.interface.configuration import load_config
from src.inverse_model.protocols import inversion_protocol_factory


def main():
    config = load_config()
    db = config.database()

    experiment_id = 0
    experiment_params = db.experiments[experiment_id]

    # EXTRA: For saving the full table of RMSEs in the paper
    nb_dss = len(experiment_params.dataset_ids)
    nb_ifms = len(experiment_params.interferometer_ids)
    nb_nls = len(experiment_params.noise_level_indices)
    nb_ips = len(experiment_params.inversion_protocol_ids)

    # EXTRA: For saving the full table of RMSEs in the paper
    reports_dir = config.directory_paths.reports
    experiment_dir = reports_dir / f"experiment_{experiment_id}"
    if not experiment_dir.exists():
        experiment_dir.mkdir(parents=True, exist_ok=True)
    full_table = np.zeros(shape=(nb_dss * nb_ips, nb_ifms * nb_nls * 2))

    for i_ds, ds_id in enumerate(experiment_params.dataset_ids):
        spectra_ref = db.dataset_spectrum(ds_id=ds_id)
        print(f"Dataset: {db.datasets[ds_id].title.upper()}")

        # EXTRA: For saving the results: Lambda_min, RMSE, Reconstructed Spectra
        dataset_dir = experiment_dir / f"invert_{db.datasets[ds_id].title}"
        if not dataset_dir.exists():
            dataset_dir.mkdir(parents=True, exist_ok=True)
        lambda_min_table = np.zeros(shape=(nb_ifms, nb_ips, nb_nls))
        rmse_table = np.zeros(shape=(nb_ifms, nb_ips, nb_nls*2))
        spectra_rec_table = np.zeros(shape=(nb_ifms, nb_ips, nb_nls, *spectra_ref.data.shape))

        for i_ifm, ifm_id in enumerate(experiment_params.interferometer_ids):
            interferometer = db.interferometer(ifm_id=ifm_id)
            print(f"\tInterferometer: {db.interferometers[ifm_id].title.upper()}")

            # Generate the transmittance response (aka Transfer Matrix)
            transfer_matrix = interferometer.transmittance_response(wavenumbers=spectra_ref.wavenumbers)

            # Simulate the interferograms (aka the data / observation / acquisition)
            interferograms = simulate_interferogram(transmittance_response=transfer_matrix, spectrum=spectra_ref)

            # Preprocessing, normalizing the transfer matrix (linear operator) and the data
            transfer_matrix = transfer_matrix.rescale(new_max=1, axis=None)
            interferograms = interferograms.rescale(new_max=1, axis=-2)

            for i_nl, nl_idx in enumerate(experiment_params.noise_level_indices):
                snr_db = db.noise_levels[nl_idx]
                print(f"\t\tSNR: {snr_db} dB")

                # Add noise to the interferogram
                np.random.seed(0)
                interferograms_noisy = interferograms.add_noise(snr_db=snr_db)

                for i_ip, ip_id in enumerate(experiment_params.inversion_protocol_ids):
                    ip_schema = db.inversion_protocols[ip_id]
                    lambdaas = ip_schema.lambdaas_schema.as_array()
                    print(f"\t\t\tSNR: {ip_schema.title.upper()}")

                    # EXTRA: For selecting the Reconstructed Spectra our of all lambdaas with the minimum RMSE
                    spectra_rec_lambdaas = np.zeros(shape=(lambdaas.size, *spectra_ref.data.shape))

                    for i_lmd, lambdaa in enumerate(lambdaas):
                        ip_kwargs = ip_schema.parameters(lambdaa=lambdaa)
                        inverter = inversion_protocol_factory(option=ip_schema.type, parameters=ip_kwargs)
                        spectra_rec = inverter.reconstruct_spectrum(
                            interferogram=interferograms_noisy, transmittance_response=transfer_matrix
                        )

                        # EXTRA: Store the spectral data array with respect to lambdaa
                        spectra_rec_lambdaas[i_lmd] = spectra_rec.data

                    # EXTRA: Store the results with minimum RMSE
                    rmse_lambdaas = calculate_rmse(
                        array=spectra_rec_lambdaas,
                        reference=spectra_ref.data[np.newaxis],
                        is_match_axis=-2,
                        is_match_stats=True,
                        is_rescale_reference=True,
                    )
                    argmin = np.argmin(rmse_lambdaas)
                    lambda_min_table[i_ifm, i_ip, i_nl] = lambdaas[argmin]
                    rmse_table[i_ifm, i_ip, i_nl] = rmse_lambdaas[argmin]
                    spectra_rec_table[i_ifm, i_ip, i_nl] = spectra_rec_lambdaas[argmin]
                    print(f"\t\t\t\tLambda: {lambda_min_table[i_ifm, i_ip, i_nl]:.3f}")
                    print(f"\t\t\t\tRMSE:   {rmse_table[i_ifm, i_ip, i_nl]:.3f}")

                    # EXTRA: Store the results with minimum RMSE in the full table for the paper
                    full_table[i_ip + nb_ips*i_ds, 2*i_nl + 2*nb_nls*i_ifm] = lambdaas[argmin]
                    full_table[i_ip + nb_ips*i_ds, 2*i_nl + 2*nb_nls*i_ifm + 1] = rmse_lambdaas[argmin]

        # EXTRA: Save the results with minimum RMSE
        file_name_list = ["lambdaa_min", "rmse", "rec_spectra"]
        table_list = [lambda_min_table, rmse_table, spectra_rec_table]
        for file_name, table in zip(file_name_list, table_list):
            np.save(file=dataset_dir / file_name, arr=table)

    # EXTRA: Save the full table for the paper
    file_name = "full_table"
    np.save(file=experiment_dir / file_name, arr=full_table)
    pprint(full_table)


if __name__ == "__main__":
    main()
