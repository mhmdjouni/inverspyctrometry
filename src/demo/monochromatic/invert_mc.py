from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt

from src.interface.configuration import load_config
from src.inverse_model.protocols import inversion_protocol_factory


def main():
    config = load_config()
    db = config.database()

    experiment_id = 1
    experiment_params = db.experiments[experiment_id]
    pprint(dict(experiment_params))

    for ds_id in experiment_params.dataset_ids:
        interferograms_ref = db.dataset_interferogram(ds_id=ds_id)
        print(f"Dataset: {db.datasets[ds_id].title.upper()}")

        wavenumbers = np.load(db.datasets[ds_id].wavenumbers_path)

        for char_id in experiment_params.interferometer_ids:
            characterization = db.characterization(char_id=char_id)
            print(f"\tCharacterization: {db.characterizations[char_id].title.upper()}")

            transfer_matrix = characterization.transmittance_response(wavenumbers=wavenumbers)
            transfer_matrix = transfer_matrix.rescale(new_max=1, axis=None)
            interferograms = interferograms_ref.rescale(new_max=1, axis=-2)

            for ip_id in experiment_params.inversion_protocol_ids:
                ip_schema = db.inversion_protocols[ip_id]
                lambdaas = ip_schema.lambdaas_schema.as_array()
                print(f"\t\t\tInvProtocol: {ip_schema.title.upper()}")

                for lambdaa in lambdaas:
                    ip_kwargs = ip_schema.ip_kwargs(lambdaa=lambdaa)
                    inverter = inversion_protocol_factory(option=ip_schema.type, ip_kwargs=ip_kwargs)
                    spectra_rec = inverter.reconstruct_spectrum(
                        interferogram=interferograms, transmittance_response=transfer_matrix
                    )

                    fig, axs = plt.subplots(nrows=1, ncols=1, squeeze=False)
                    spectra_rec.visualize_matrix(
                        axs=axs[0, 0],
                        vmin=None,
                        vmax=None,
                    )
                    plt.show()


if __name__ == "__main__":
    main()
