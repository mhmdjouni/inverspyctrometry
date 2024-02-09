from pprint import pprint

import numpy as np

from src.interface.configuration import load_config


def main():
    config = load_config()
    db = config.database()

    reports_folder = config.directory_paths.reports

    experiment_id_options = [2]

    for experiment_id in experiment_id_options:
        experiment_params = db.experiments[experiment_id]
        pprint(dict(experiment_params))
        experiment_dir = reports_folder / f"experiment_{experiment_id}"

        for ds_id in experiment_params.dataset_ids:
            interferograms_ref = db.dataset_interferogram(ds_id=ds_id)
            wavenumbers_ifgm = db.dataset_central_wavenumbers(dataset_id=ds_id)
            print(f"Dataset: {db.datasets[ds_id].title.upper()}")
            dataset_dir = experiment_dir / f"invert_{db.datasets[ds_id].title}"

            interferograms_ref = interferograms_ref.rescale(new_max=1, axis=-2)

            for char_id in experiment_params.interferometer_ids:
                characterization = db.characterization(char_id=char_id)
                print(f"\tCharacterization: {db.characterizations[char_id].title.upper()}")
                characterization_dir = dataset_dir / f"{db.characterizations[char_id].title}"

                transfer_matrix = characterization.transmittance_response(wavenumbers=wavenumbers_ifgm)
                transfer_matrix = transfer_matrix.rescale(new_max=1, axis=None)

                for ip_id in experiment_params.inversion_protocol_ids:
                    lambdaas = db.inversion_protocol_lambdaas(inv_protocol_id=ip_id)
                    print(f"\t\tInversion Protocol: {db.inversion_protocols[ip_id].title.upper()}")
                    inverter_dir = characterization_dir / f"{db.inversion_protocols[ip_id].title}"

                    spectra_rec_all = np.zeros(shape=(lambdaas.size, wavenumbers_ifgm.size, interferograms_ref.data.shape[-1]))

                    for il, lambdaa in enumerate(lambdaas):
                        inverter = db.inversion_protocol(
                            inv_protocol_id=ip_id,
                            lambdaa=lambdaa,
                            is_compute_and_save_cost=False,
                            experiment_id=-1,
                        )
                        spectra_rec = inverter.reconstruct_spectrum(
                            interferogram=interferograms_ref, transmittance_response=transfer_matrix
                        )
                        spectra_rec_all[il] = spectra_rec.data

                    # Save all spectral reconstructions wrt lambda, per inv_protocol per dataset
                    if not inverter_dir.exists():
                        inverter_dir.mkdir(parents=True, exist_ok=True)
                    np.save(file=inverter_dir / "spectra_rec_all", arr=spectra_rec_all)

            # Save the wavenumbers for the sake of independent plots
            if not dataset_dir.exists():
                dataset_dir.mkdir(parents=True, exist_ok=True)
            np.save(file=dataset_dir / "wavenumbers", arr=wavenumbers_ifgm)


if __name__ == "__main__":
    main()
