"""
Minimal working template for simulated inversion, without comments and extra functionality (saving arrays, etc.)
"""

import numpy as np
from matplotlib import pyplot as plt

from src.direct_model.interferometer import simulate_interferogram
from src.interface.configuration import load_config
from src.inverse_model.protocols import inversion_protocol_factory


def main():
    """
    - Load the dataset
    - Load the interferometer
    - Generate the transmittance response (i.e., Transfer Matrix)
    - Simulate the interferograms (i.e., the data, observation, acquisition)
    - Preprocessing: Rescaling / Normalizing the transfer matrix (linear operator) and the data
    - Add noise to the interferogram
    - Load the inversion protocol
    - Inversion: Reconstruct the spectrum
    """

    config = load_config()
    db = config.database()

    experiment_id = 0
    experiment_params = db.experiments[experiment_id]

    for ds_id in experiment_params.dataset_ids:
        spectra_ref = db.dataset_spectrum(ds_id=ds_id)
        print(f"Dataset: {db.datasets[ds_id].title.upper()}")

        for ifm_id in experiment_params.interferometer_ids:
            interferometer = db.interferometer(ifm_id=ifm_id)
            print(f"\tInterferometer: {db.interferometers[ifm_id].title.upper()}")

            transfer_matrix = interferometer.transmittance_response(wavenumbers=spectra_ref.wavenumbers)
            interferograms = simulate_interferogram(transmittance_response=transfer_matrix, spectrum=spectra_ref)
            transfer_matrix = transfer_matrix.rescale(new_max=1, axis=None)
            interferograms = interferograms.rescale(new_max=1, axis=-2)

            for nl_idx in experiment_params.noise_level_indices:
                snr_db = db.noise_levels[nl_idx]
                print(f"\t\tSNR: {snr_db} dB")

                np.random.seed(0)
                interferograms_noisy = interferograms.add_noise(snr_db=snr_db)

                for ip_id in experiment_params.inversion_protocol_ids:
                    ip_schema = db.inversion_protocols[ip_id]
                    lambdaas = ip_schema.lambdaas_schema.as_array()
                    print(f"\t\t\tSNR: {ip_schema.title.upper()}")

                    for lambdaa in lambdaas:
                        ip_kwargs = ip_schema.ip_kwargs(lambdaa=lambdaa)
                        inverter = inversion_protocol_factory(option=ip_schema.type, ip_kwargs=ip_kwargs)
                        spectra_rec = inverter.reconstruct_spectrum(
                            interferogram=interferograms_noisy, transmittance_response=transfer_matrix
                        )

                        fig, axs = plt.subplots(nrows=1, ncols=1, squeeze=False)
                        spectra_rec.visualize(axs=axs[0, 0], acq_ind=0)
                        plt.show()


if __name__ == "__main__":
    main()
