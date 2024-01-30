from pprint import pprint

import numpy as np
from tqdm import tqdm

from src.direct_model.interferometer import simulate_interferogram
from src.interface.configuration import load_config
from src.inverse_model.protocols import inversion_protocol_factory


def main():
    db = load_config().database()

    experiment_params = db.experiments[0]

    for ds_id in experiment_params.dataset_ids:
        spectra_ref = db.dataset_spectrum(ds_id=ds_id)

        for ifm_id in experiment_params.interferometer_ids:
            interferometer = db.interferometer(ifm_id=ifm_id)
            transfer_matrix = interferometer.transmittance_response(wavenumbers=spectra_ref.wavenumbers)
            interferograms = simulate_interferogram(transmittance_response=transfer_matrix, spectrum=spectra_ref)
            transfer_matrix = transfer_matrix.rescale(new_max=1, axis=None)
            interferograms = interferograms.rescale(new_max=1, axis=-2)

            for snr_idx in experiment_params.noise_level_indices:
                snr_db = db.noise_levels[snr_idx]
                interferograms_noisy = interferograms.add_noise(snr_db=snr_db)

                for ip_id in experiment_params.inversion_protocol_ids:
                    ip_schema = db.inversion_protocols[ip_id]

                    for lambdaa in tqdm(ip_schema.lambdaas_schema.as_array()):
                        ip_kwargs = ip_schema.ip_kwargs(lambdaa=lambdaa)
                        inverter = inversion_protocol_factory(option=ip_schema.type, ip_kwargs=ip_kwargs)
                        spectra_rec = inverter.reconstruct_spectrum(
                            interferogram=interferograms_noisy, transmittance_response=transfer_matrix
                        )

                    # print("Computing the RMSE of the reconstruction vs reference spectra w.r.t lambdaas...")
                    # print("Selecting lambdaa_min, and the associated RMSE and reconstructed spectrum...")
                    print(f"\nFinished Inversion Protocol {ip_schema.title.upper()}\n")

                print(f"\nFinished SNR = {snr_db} dB\n")


if __name__ == "__main__":
    main()
